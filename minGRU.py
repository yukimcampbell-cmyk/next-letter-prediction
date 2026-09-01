import torch
import torch.nn as nn
import matplotlib.pylab as plt
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class minGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                outfcn=nn.LogSoftmax(dim=-1)):
        super().__init__()

        self.in_dim = in_dim 
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim 

        self.h = None

        # Learnable transformations for the update gate, candidate
        # hidden state, and output.
        self.gate = nn.Linear(in_dim, hidden_dim)
        self.candidate = nn.Linear(in_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)

        self.sigmoid = nn.Sigmoid()
        self.outfcn = outfcn

        self.losses = []
        self.to(device)

    def reset(self):
        self.h = None

    def step(self, x):
        _, T, _ = x.shape
        assert T == 1

        if self.h is None:
            self.h = torch.zeros(
                (x.shape[0], 1, self.hidden_dim)
                )

        # The gate determines how much of the new candidate state
        # should replace the previous hidden state. 
        g = self.sigmoid(self.gate(x[:,0,:]))

        # Compute the candidate hidden state from the previous timestep.
        h_tilde = self.candidate(x[:,0,:])

        # Retrieve the hidden state from the previous timestep.
        h_prev = self.h[:,0,:]

        # Interpolate between the previous and candidate hidden states.
        h_new = g * h_tilde + (1 - g) * h_prev

        self.h = h_new[:,None,:]

        # Convert the hidden state into the output prediction.
        y = self.outfcn(self.output(h_new))[:,None,:]
        return y

    def forward(self, x_batch):
        samples, T, in_dim = x_batch.shape
        assert in_dim == self.in_dim

        x_batch = x_batch.to(device)
        self.reset()

        # Store the output prediction from each timestep.
        y = torch.zeros(samples, T, self.out_dim, device=device)

        # Process the sequence one character at a time, carrying the 
        # hidden state between timesteps.
        for t in range(T):
          x_t = x_batch[:,t:t+1,:]
          y_t = self.step(x_t)
          y[:,t,:] = y_t[:,0,:]

        return y

class DeepminGRU(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.losses = []

        # Add multiple minGRU layers to create a deeper recurrent model.
        self.layers = nn.ModuleList()

        for l in range(n_layers):
            input_size = in_dim if l == 0 else hidden_dim

            # The final layer produces a probability distribution over the 
            # character vocabulary.
            if l == n_layers - 1:
                out_dim = in_dim
                outfcn = nn.LogSoftmax(dim=-1)
            else:
                out_dim = hidden_dim
                outfcn = nn.Identity()

            self.layers.append(
                minGRU(
                    input_size,
                    hidden_dim,
                    out_dim,
                    outfcn=outfcn
                )
            )

        self.to(device)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def step(self, x):
        for layer in self.layers:
          x = layer.step(x)
        return x

    def forward(self, x):
        self.reset()
        batch, T, _ = x.shape
        out_seq = torch.zeros(x.size(0), T, self.in_dim, device=device)
        for t in range(T):
          out = x[:,t:t+1,:]
          for layer in self.layers:
            out = layer.step(out)
          out_seq[:,t,:] = out[:,0,:]
        x = out_seq
        return x

    def predict(self, x, n=10):
        with torch.no_grad():
            assert x.shape[0] == 1
            assert x.shape[2] == self.in_dim
            output = torch.zeros((1, n, self.in_dim)).to(device)
            self.reset()
            self.forward(x)

            inp = x[:, -1:, :]

            for t in range(n):
                y_t = self.step(inp)
                output[:,t,:] = y_t[:,0,:]
                idx = y_t[:,0,:].argmax(dim=1)
                inp = torch.zeros((1, 1, self.in_dim)).to(device)
                inp[0, 0, idx.item()] = 1.0

        return output

    def bptt(self, dl, epochs=10, lr=0.001):
        # Negative log-likelihood loss is a fit for the LogSoftmax
        # outputs of the final layer.
        loss_fcn = nn.NLLLoss()

        # Adam updates the model parameters using gradients.
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            total_loss = 0.

            for x,t in (dl):
                # Forward pass.
                y = self(x)

                # Add up prediction loss across all timesteps.
                loss = torch.tensor(0., device=device, requires_grad=True)

                for k in range(t.shape[1]):
                    ys = y[:,k,:].squeeze()
                    ts = t[:,k]
                    loss = loss + loss_fcn(ys, ts)

                # Backpropagate and update weights.
                optim.zero_grad()
                loss.backward()
                optim.step()

                total_loss += loss.detach().cpu().item() * len(t)

            self.losses.append(total_loss/len(dl.dataset))

        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title("Training Loss")
        plt.show()