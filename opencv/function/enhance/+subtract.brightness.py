import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "../img/coast.jpg")
img = cv2.imread(path, cv2.IMREAD_COLOR)

matrix = np.ones(img.shape, dtype="uint8") * 50

img_rgb_brighter = cv2.add(img, matrix)
img_rgb_darker = cv2.subtract(img, matrix)


# Show the images
plt.figure(figsize=[18, 5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Darker");
plt.subplot(132); plt.imshow(img);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Brighter");
plt.show()