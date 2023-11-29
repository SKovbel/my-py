import torch
import torch.nn as nn

input = torch.tensor([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]])
layer = nn.Flatten(0, -1)
layer = nn.Flatten(0, 1)
output = layer(input)

print(input, output)
