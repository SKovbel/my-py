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

dir = os.path.join(os.getcwd(), f"../../tmp/opencv/4")
if not os.path.exists(dir):
    os.mkdir(dir)

URL = r"https://www.dropbox.com/s/0oe92zziik5mwhf/opencv_bootcamp_assets_NB4.zip?dl=1"
asset_zip_path = os.path.join(dir, "opencv_bootcamp_assets_NB4.zip")
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)


plt.figure(figsize=[177, 7])


# COCA
img_bgr = cv2.imread(f"{dir}/coca-cola-logo.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.subplot(171);plt.imshow(img_rgb)
logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]



img_background_bgr = cv2.imread(f"{dir}/checkerboard_color.png")
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)
aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)
plt.subplot(172);plt.imshow(img_background_rgb)



img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
plt.subplot(173);plt.imshow(img_mask, cmap="gray")



img_mask_inv = cv2.bitwise_not(img_mask)
plt.subplot(174);plt.imshow(img_mask_inv, cmap="gray")



img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)
plt.subplot(175);plt.imshow(img_background)


img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
plt.subplot(176);plt.imshow(img_foreground)


result = cv2.add(img_background, img_foreground)
plt.subplot(177);plt.imshow(result)
cv2.imwrite("logo_final.png", result[:, :, ::-1])
plt.show()



