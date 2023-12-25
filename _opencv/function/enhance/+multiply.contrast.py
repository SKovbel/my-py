import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "../img/coast.jpg")
img = cv2.imread(path, cv2.IMREAD_COLOR)

matrix1 = np.ones(img.shape) * 0.8
matrix2 = np.ones(img.shape) * 1.2

img_rgb_darker = np.uint8(cv2.multiply(np.float64(img), matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img), matrix2))


# Show the images
plt.figure(figsize=[18, 5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Darker");
plt.subplot(132); plt.imshow(img);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Brighter");
plt.show()
