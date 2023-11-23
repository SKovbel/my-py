import numpy as np

# Creating an array with a singleton dimension
arr_with_singleton_dim = np.array([[1], [2], [3]])

# Using squeeze to remove the singleton dimension
arr_squeezed = np.squeeze(arr_with_singleton_dim)

# Displaying the original and squeezed arrays
print("Original array:")
print(arr_with_singleton_dim)
print("Squeezed array:")
print(arr_squeezed)
