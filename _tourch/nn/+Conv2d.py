import torch
import torch.nn as nn

input = torch.tensor([[
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
]])

layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1)
output = layer(input)
print(input, output)
