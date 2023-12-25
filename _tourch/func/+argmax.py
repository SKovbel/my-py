import torch
# The torch.argmax function is used to find the indices of the maximum values along a specified axis
tensor = torch.tensor([[1, 3, 2], [6, 5, 4], [7, 8, 9]])

max_indices_row = torch.argmax(tensor, dim=1)
max_indices_col = torch.argmax(tensor, dim=0)

print(tensor)
print(max_indices_row)
print(max_indices_col)
