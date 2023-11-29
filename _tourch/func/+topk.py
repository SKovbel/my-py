import torch
# The torch.topk function is used to find the k largest elements along a specified axis.
tensor = torch.tensor([[1, 5, 3], [4, 2, 6], [7, 0, 9]])
print(tensor)

top_values, top_indices = torch.topk(tensor, k=2, dim=1)
print('case1')
print(top_values)
print(top_indices)


top_values, top_indices = torch.topk(tensor, k=1, dim=1)
print('case2')
print(top_values)
print(top_indices)

top_values, top_indices = torch.topk(tensor, k=1, dim=0)
print('case3')
print(top_values)
print(top_indices)
