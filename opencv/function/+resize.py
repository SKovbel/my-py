#https://courses.opencv.org/courses/course-v1:OpenCV+Bootcamp+CV0/courseware/457799bde2064b749df7fb0c0a741b5f/6f72f10a1dbe41639e8a49094f92b53d/1?activate_block_id=block-v1%3AOpenCV%2BBootcamp%2BCV0%2Btype%40vertical%2Bblock%406c60f376f660464f9631f4174d3b6256

import os
import cv2
import matplotlib.pyplot as plt

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "img/coca-cola-logo.png")
img = cv2.imread(path, cv2.IMREAD_COLOR)

img2 = cv2.resize(img, None, fx=2, fy=2)

plt.imshow(img2)

# Medthod 2: Specifying exact size of the output image
desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)
img3 = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(img3)

plt.show()

#Resize while maintaining aspect ratio
cropped_region = img[200:400, 300:600]
desired_width = 100
aspect_ratio = desired_width / cropped_region.shape[1]
desired_height = int(cropped_region.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)

# Resize image
resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
