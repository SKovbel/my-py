'''
cv2.cvtColor() Converts an image from one color space to another.
The function converts an input image from one color space to another.
In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR).
Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed).
So the first byte in a standard (24-bit) color image will be an 8-bit Blue component,
the second byte will be Green, and the third byte will be Red.
The fourth, fifth, and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
'''

import os
import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=[20,5])

path = os.path.join(os.getcwd(), "img/coca-cola-logo.png")
img1 = cv2.imread(path, cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

plt.subplot(131);plt.imshow(img1)
plt.subplot(132);plt.imshow(img2)
plt.subplot(133);plt.imshow(img3)
plt.show()
