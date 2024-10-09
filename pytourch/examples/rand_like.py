import torch

# Create a tensor with specific values
input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Generate a tensor with the same shape as input_tensor, filled with random numbers
random_tensor = torch.randn_like(input_tensor)

print("Input tensor:")
print(input_tensor)
print("\nRandom tensor with the same shape:")
print(random_tensor)
