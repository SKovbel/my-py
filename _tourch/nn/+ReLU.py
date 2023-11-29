import torch
import torch.nn as nn

input = torch.tensor([[1.0, -2.0, 3.0]])
layer = nn.ReLU()
output = layer(input)

print(input, output)
