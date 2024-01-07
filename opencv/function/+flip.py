#https://courses.opencv.org/courses/course-v1:OpenCV+Bootcamp+CV0/courseware/457799bde2064b749df7fb0c0a741b5f/6f72f10a1dbe41639e8a49094f92b53d/1?activate_block_id=block-v1%3AOpenCV%2BBootcamp%2BCV0%2Btype%40vertical%2Bblock%406c60f376f660464f9631f4174d3b6256

import os
import cv2
import matplotlib.pyplot as plt

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1

path = os.path.join(os.getcwd(), "img/coca-cola-logo.png")
img = cv2.imread(path, cv2.IMREAD_COLOR)



img_NZ_rgb_flipped_horz = cv2.flip(img, 1)
img_NZ_rgb_flipped_vert = cv2.flip(img, 0)
img_NZ_rgb_flipped_both = cv2.flip(img, -1)

# Show the images
plt.figure(figsize=(18, 5))
plt.subplot(141);plt.imshow(img_NZ_rgb_flipped_horz);plt.title("Horizontal Flip");
plt.subplot(142);plt.imshow(img_NZ_rgb_flipped_vert);plt.title("Vertical Flip");
plt.subplot(143);plt.imshow(img_NZ_rgb_flipped_both);plt.title("Both Flipped");
plt.subplot(144);plt.imshow(img);plt.title("Original");

plt.show()