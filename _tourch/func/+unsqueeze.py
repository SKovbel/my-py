import torch

# Example tensor
original_tensor = torch.tensor([1, 2, 3, 4])

# Unsqueezing along dimension 0 (adding a batch dimension)
unsqueezed_tensor_batch = original_tensor.unsqueeze(0)

# Unsqueezing along dimension 1 (adding a channel dimension)
unsqueezed_tensor_channel = original_tensor.unsqueeze(1)

# Print the original and unsqueezed tensors
print("Original Tensor:", original_tensor)
print("Unsqueezed Tensor (Batch Dimension):", unsqueezed_tensor_batch)
print("Unsqueezed Tensor (Channel Dimension):", unsqueezed_tensor_channel)
