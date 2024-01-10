import numpy as np

# Create a numpy array
arr = np.array([1, 2, 3, 4, 5])

# Use numpy.where() to create a new array based on a condition
new_arr = np.where(arr < 3, 0, 1)
new_arr2 = np.where([arr >= 3])

# Print the original and new arrays
print("Original Array:", arr)
print("New Array:", new_arr)
print("New Array2 indices:", new_arr2[1])
