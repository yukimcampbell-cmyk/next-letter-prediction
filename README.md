# next-letter-prediction

A PyTorch implementation of a 3-layer minGRU trained to predict the next character in Pride and Prejudice. The model learns character-level patterns from the text and uses previous characters to predict the next character in the sequence.

## Tools:
- Python
- PyTorch
- NumPy
- Matplotlib

## Model & Training:
- 3-layer minGRU
- 300 hidden units per layer
- Character vocabulary of 27 characters
- Sequence length: 15
- Adam optimizer
- Backpropagation through time (BPTT)
- 80 training epochs

## Results:
**- 94.0% character accuracy**
- Training loss tracked through epochs
- Autoregressive text generation implemented using the trained model

## Experiments:
The model was evaluated across different sequence lengths and network configurations to investigate how historical context and model depth affect next-character prediction.
