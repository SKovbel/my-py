
import os
import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=[20,5])

path = os.path.join(os.getcwd(), "img/coca-cola-logo.png")
img = cv2.imread(path, cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)

img1 = cv2.merge((b, g, r))
img2 = cv2.merge((b, r, g))
img3 = cv2.merge((g, r, r))

plt.subplot(131);plt.imshow(img1)
plt.subplot(132);plt.imshow(img2)
plt.subplot(133);plt.imshow(img3)
plt.show()

