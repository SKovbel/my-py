import torch
import torch.nn as nn

input = torch.tensor([[1.0, -2.0, 3.0]])

# wx+b
layer = nn.Linear(3, 3)
output = layer(input)
print(input, output)