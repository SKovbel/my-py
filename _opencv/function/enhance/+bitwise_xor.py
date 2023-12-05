import os
import cv2
import matplotlib.pyplot as plt

# OpenCV Documentation
img_rec = cv2.imread(f"../img/rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread(f"../img/circle.jpg", cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=[20, 5])
plt.subplot(121);plt.imshow(img_rec, cmap="gray")
plt.subplot(122);plt.imshow(img_cir, cmap="gray")

result = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray")
plt.show()
