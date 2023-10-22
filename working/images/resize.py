import cv2

image = cv2.imread('vg_night.png')
resized_image = cv2.resize(image, (640, 480))
cv2.imwrite('vg_night_640_x_480.png', resized_image)


image = cv2.imread('win_xp.png')
resized_image = cv2.resize(image, (640, 480))
cv2.imwrite('win_xp_640_x_480.png', resized_image)
