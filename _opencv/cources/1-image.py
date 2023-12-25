import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from zipfile import ZipFile
from urllib.request import urlretrieve
from IPython.display import Image, display


def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")
    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

dir= os.path.join(os.getcwd(), f"../../tmp/opencv")
URL = r"https://www.dropbox.com/s/qhhlqcica1nvtaw/opencv_bootcamp_assets_NB1.zip?dl=1"
asset_zip_path = os.path.join(dir, "opencv_bootcamp_assets_NB1.zip")
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# Display 18x18 pixel image.
png = os.path.join(dir, "checkerboard_18x18.png")
display(Image(filename=os.path.join(dir, "checkerboard_18x18.png")))

# cv2.IMREAD_GRAYSCALE or 0: Loads image in grayscale mode
# cv2.IMREAD_COLOR or 1: Loads a color image. Any transparency of image will be neglected. It is the default flag.
# cv2.IMREAD_UNCHANGED -1
cb_img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)

#cv2.imshow("Image", cb_img )
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print(cb_img)
print("Image size (H, W) is:", cb_img.shape)
print("Data type of image is:", cb_img.dtype)

plt.imshow(cb_img)
#plt.show()

plt.imshow(cb_img, cmap="gray")
#plt.show()


# Read image as gray scale.
cb_img_fuzzy = cv2.imread(os.path.join(dir, "checkerboard_fuzzy_18x18.jpg"), 0)
print(cb_img_fuzzy)
plt.imshow(cb_img_fuzzy, cmap="gray")
#plt.show()


# Read in image
coke_img = cv2.imread(os.path.join(dir, "coca-cola-logo.png"), 1)
print("Image size (H, W, C) is:", coke_img.shape)
print("Data type of image is:", coke_img.dtype)
plt.imshow(coke_img)
#plt.show()

#inverse color
coke_img_channels_reversed = coke_img[:, :, ::-1]
plt.imshow(coke_img_channels_reversed)
#plt.show()

#Splitting and Merging Color Channels
# Split the image into the B,G,R components
img_NZ_bgr = cv2.imread(os.path.join(dir, "New_Zealand_Lake.jpg"), cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)

# Show the channels
plt.figure(figsize=[155, 5])

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel")
imgMerged = cv2.merge((b, g,  r))
plt.imshow(imgMerged[:, :, ::-1])
plt.title("Merged Output")
#plt.show()


img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_NZ_rgb)


img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)
# Split the image into the B,G,R components
h,s,v = cv2.split(img_hsv)
# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb);   plt.title("Original");
#plt.show()

#Changing to HSV color space
img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)
# Split the image into the B,G,R components
h,s,v = cv2.split(img_hsv)


img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)

# Split the image into the B,G,R components
h,s,v = cv2.split(img_hsv)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb);   plt.title("Original");

# Modifying individual Channel
h_new = h + 10
img_NZ_merged = cv2.merge((h_new, s, v))
img_NZ_rgb = cv2.cvtColor(img_NZ_merged, cv2.COLOR_HSV2RGB)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb);   plt.title("Original");

cv2.imwrite("New_Zealand_Lake_SAVED.png", img_NZ_bgr)
Image(filename='New_Zealand_Lake_SAVED.png')