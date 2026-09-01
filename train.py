import torch
import pp_dataset
from minGRU import DeepminGRU

# Load the character-level dataset.
pp = pp_dataset.Origin(
  start=41,
  stop=10015+41,
  seq_length=15
)

dl = torch.utils.data.DataLoader(
  pp, batch_size=128,
  shuffle=True
)

# Create a 3-layer minGRU with 300 hidden neurons.
net = DeepminGRU(
    n_layers=3,
    in_dim=27,
    hidden_dim=300
)

# Train the model using backpropagation through time.
net.bptt(
    dl,
    epochs=80,
    lr=0.001
)

# Evaluate next-character prediction accuracy.
correct = 0
total = 0

for x, t in dl:
  net.reset()
  y = net(x)

  prediction = y[:,-1,:].argmax(dim=1)
  target = t[:,-1]

  correct += (prediction == target).sum().item()
  total += t.shape[0]

accuracy = correct / total

print(f"Next-character accuracy: {accuracy:.1%}")
