import torch
import torch.nn as nn

# Example input tensor
input_tensor = torch.tensor([[1.0, -2.0, 3.0]])

# Create a ReLU activation layer
relu_layer = nn.ReLU()

# Apply the ReLU activation
output_tensor = relu_layer(input_tensor)

# Print the result
print(output_tensor.numpy())
