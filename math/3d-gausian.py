import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 30  # Size of each dimension of the array
num_elements = N**3  # Total number of elements in the array
num_non_zero = int(0.1 * num_elements)  # 10% non-zero elements

# Create a 3D array filled with zeros
array_3d = np.zeros((N, N, N))

# Flatten the array to easily select random indices
flat_array = array_3d.flatten()

# Randomly select 10% of the indices to be non-zero
non_zero_indices = np.random.choice(np.arange(num_elements), num_non_zero, replace=False)

# Assign random values from 1 to 8 to the selected indices
flat_array[non_zero_indices] = np.random.randint(1, 9, num_non_zero)

# Reshape the array back to 3D
array_3d = flat_array.reshape((N, N, N))
print(array_3d)
# Create a Gaussian distribution array
mean = N // 2  # Center of the Gaussian distribution
std_dev = 5  # Standard deviation of the Gaussian distribution

# Define the Gaussian function
def gaussian(x, y, z, mean, std_dev):
    return np.exp(-((x - mean)**2 + (y - mean)**2 + (z - mean)**2) / (2 * std_dev**2))

# Create a 3D grid for the Gaussian distribution
x, y, z = np.meshgrid(np.arange(N), np.arange(N), np.arange(N))
gaussian_distribution = gaussian(x, y, z, mean, std_dev)

# Normalize the Gaussian distribution to the range [1, 8]
gaussian_distribution = 0 + 8 * (gaussian_distribution - gaussian_distribution.min()) / (gaussian_distribution.max() - gaussian_distribution.min())
print(gaussian_distribution)

# Visualize the 3D array with random values
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(array_3d[N//2, :, :] if i == 0 else (array_3d[:, N//2, :] if i == 1 else array_3d[:, :, N//2]),
              cmap='viridis', interpolation='none')
    ax.set_title(['XY Slice', 'XZ Slice', 'YZ Slice'][i])
plt.suptitle('3D Array with Random Values from 1 to 8')
plt.show()

# Visualize the Gaussian distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(gaussian_distribution[N//2, :, :] if i == 0 else (gaussian_distribution[:, N//2, :] if i == 1 else gaussian_distribution[:, :, N//2]),
              cmap='viridis', interpolation='none')
    ax.set_title(['XY Slice', 'XZ Slice', 'YZ Slice'][i])
plt.suptitle('Gaussian Distribution')
plt.show()

# 3D scatter plot of the non-zero values in the 3D array
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
non_zero_indices = np.argwhere(array_3d > 0)
ax.scatter(non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2],
           c=array_3d[non_zero_indices[:, 0], non_zero_indices[:, 1], non_zero_indices[:, 2]],
           cmap='viridis')
ax.set_title('3D Scatter Plot of Non-Zero Values')
plt.show()

# 3D scatter plot of the Gaussian distribution
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
gaussian_indices = np.argwhere(gaussian_distribution > 0)
ax.scatter(gaussian_indices[:, 0], gaussian_indices[:, 1], gaussian_indices[:, 2],
           c=gaussian_distribution[gaussian_indices[:, 0], gaussian_indices[:, 1], gaussian_indices[:, 2]],
           cmap='viridis')
ax.set_title('3D Scatter Plot of Gaussian Distribution')
plt.show()
