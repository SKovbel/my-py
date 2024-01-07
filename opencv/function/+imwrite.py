import os
import cv2

path1 = os.path.join(os.getcwd(), "img/coca-cola-logo.png")
path2 = os.path.join(os.getcwd(), "img/coca-cola-logo_gray.png")

img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

# read the image as gray scaled
cv2.imwrite(path2, img)

