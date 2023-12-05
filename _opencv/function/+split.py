
import os
import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=[20,5])

path = os.path.join(os.getcwd(), "img/coca-cola-logo.png")
img = cv2.imread(path, cv2.IMREAD_COLOR)
b, g, r = cv2.split(img)

plt.subplot(131);plt.imshow(b)
plt.subplot(132);plt.imshow(g)
plt.subplot(133);plt.imshow(r)
plt.show()

