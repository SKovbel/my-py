import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load PNG images using OpenCV
def load_images(image_paths):
    images = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
        if image is not None:
            images.append(image)
    return np.array(images)

# Example: Load two PNG images
image_paths = ["image1.png", "image2.png"]
images = load_images(image_paths)

# Normalize pixel values (if needed)
images = images.astype('float32') / 255.0

# Define the AE-CNN architecture
def encoder(input_image):
    # Implement your encoder architecture here (convolutional layers, pooling, etc.)
    # Example: A simple encoder with one convolutional layer
    conv1 = np.convolve(input_image, np.random.rand(3, 3, 1, 16), 'same')
    return conv1

def decoder(encoded_image):
    # Implement your decoder architecture here (deconvolution, upsampling, etc.)
    # Example: A simple decoder with one deconvolutional layer
    deconv1 = np.convolve(encoded_image, np.random.rand(3, 3, 16, 1), 'same')
    return deconv1

# Encode the images
encoded_images = []
for image in images:
    encoded_images.append(encoder(image))
encoded_images = np.array(encoded_images)

# Decode the encoded images
decoded_images = []
for encoded_image in encoded_images:
    decoded_images.append(decoder(encoded_image))
decoded_images = np.array(decoded_images)

# Display the original and reconstructed images using Matplotlib
n = len(images)
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i], cmap='gray')
    plt.axis('off')

plt.show()
