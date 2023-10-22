import numpy as np

# Create a 1D array
arr = np.array([1, 2, 3])

# Use np.expand_dims to convert it to a 2D column vector
column_vector = np.expand_dims(arr, axis=1)

print(arr)
print(column_vector)
