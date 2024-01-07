import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "../img/coast.jpg")
img = cv2.imread(path, cv2.IMREAD_COLOR)

retval, img_thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

# Show the images
plt.figure(figsize=[18, 5])
plt.subplot(121);plt.imshow(img, cmap="gray");  plt.title("Original")
plt.subplot(122);plt.imshow(img_thresh, cmap="gray"); plt.title("Thresholded")
plt.show()
