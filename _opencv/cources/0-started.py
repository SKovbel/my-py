import cv2

# Load image
img = cv2.imread('../examples/test-img.jpg')

if img is None:
    print('Error: Image not loaded')
else:
    cv2.imshow('Window Title', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
