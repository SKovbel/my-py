import os
import cv2
import matplotlib.pyplot as plt

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "../img/coca-cola-logo.png")
img = cv2.imread(path, cv2.IMREAD_COLOR)


text = "Hello, cola"
fontScale = 2.3
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2
cv2.putText(img, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);
plt.imshow(img[:, :, ::-1])
plt.show()


text = "Hello, cola"
fontScale = -12.3
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2
cv2.putText(img, text, (500, 200), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);
plt.imshow(img[:, :, ::-1])
plt.show()

plt.imshow(img[:,:,::-1])
plt.show()
