import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def gaussian_2d(size, mu_x, mu_y, sigma_x, sigma_y):
    x = np.linspace(-3*sigma_x, 3*sigma_x, size[0])
    y = np.linspace(-3*sigma_y, 3*sigma_y, size[1])
    x, y = np.meshgrid(x, y)
    
    z = (1 / (2 * np.pi * sigma_x * sigma_y) * 
         np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + 
                  (y - mu_y)**2 / (2 * sigma_y**2))))
    return z

# Function to generate a 2D Gaussian distribution
def gaussian_2d2(size, sigma_x, sigma_y):
    x = np.linspace(-size[0] // 2, size[0] // 2, size[0])
    y = np.linspace(-size[1] // 2, size[1] // 2, size[1])
    x, y = np.meshgrid(x, y)
    
    z = np.exp(-((x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2))))
    return z

# Create a sample 2D array (image)
array = np.array([[1,1,0],
                  [1,0,0],
                  [0,0,0]])

array = array + 1
# Parameters for the Gaussian distribution
mu_x, mu_y = 0.0, 0.0      # means
sigma_x, sigma_y = 10.0, 10.0  # standard deviations

# Generate a Gaussian distribution

#gaussian = gaussian_2d(array.shape, mu_x, mu_y, sigma_x, sigma_y)
gaussian = gaussian_2d2(array.shape, sigma_x, sigma_y)
gaussian = gaussian / gaussian.max()
gaussian = array * gaussian
print(gaussian)

# Plot the original and smoothed arrays
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(array, cmap='gray')
ax[0].set_title('Original Array')
ax[1].imshow(gaussian, cmap='gray')
ax[1].set_title('Smoothed Array')

plt.show()
