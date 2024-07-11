import numpy as np
from sklearn.model_selection import train_test_split

# Define the dimensions of the images and the number of samples
num_samples = 1
img_height = 28
img_width = 28
num_colors = 7

# Generate synthetic data
# Each pixel value is randomly chosen from 1 to 7
images = np.random.randint(1, num_colors + 1, size=(num_samples, img_height, img_width))
print(images.shape)

# Normalize the pixel values (optional, depending on your network)
images_normalized = images / num_colors
print(images_normalized.shape)

# One-hot encode the images (optional, depending on your network)
images_one_hot = np.eye(num_colors)[images - 1]  # Subtract 1 to get 0-based indexing
print(images_one_hot.shape)