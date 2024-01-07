import os
import cv2
import matplotlib.pyplot as plt

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "../img/coca-cola-logo.png")
img = cv2.imread(path, cv2.IMREAD_COLOR)

cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)

plt.imshow(img[:,:,::-1])
plt.show()

cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), thickness=-5, lineType=cv2.LINE_8)

plt.imshow(img[:,:,::-1])
plt.show()