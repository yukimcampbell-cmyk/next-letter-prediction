# Next-Letter Prediction

A PyTorch implementation of a 3-layer minGRU trained to predict the next character in _Pride and Prejudice_. The model learns character-level patterns from the text and uses previous characters to predict the next character in the sequence.

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
- **94.0% character accuracy**

## Experiments:
The model was evaluated across different sequence lengths and network configurations to investigate how historical context and model depth affect next-character prediction.

To investigate how the amount of historical context affects prediction performance, I tested different sequence lengths while keeping the model architecture and training settings consistent. The experiment compares how providing the model with more previous characters affects its ability to predict the next character and generate text.
