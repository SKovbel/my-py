# Import libraries
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tarfile
from zipfile import ZipFile
from urllib.request import urlretrieve

DIR = os.path.join(os.getcwd(), f"../../tmp/opencv/13")
URL = r"https://www.dropbox.com/s/xoomeq2ids9551y/opencv_bootcamp_assets_NB13.zip?dl=1"
NET = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz'
path = lambda name: os.path.join(DIR, name)
TAR = path('ssd_mobilenet_v2_coco_2018_03_29.tar.gz')
ZIP = path('opencv_bootcamp_assets_NB13.zip')


def untar(file_path):
    file = tarfile.open(file_path)
    print(os.path.split(file_path)[0])
    file.extractall(os.path.split(file_path)[0])
    file.close()
    print("Unzipped", file_path)


def unzip(file_path):
    # Extracting zip file using the zipfile package.
    with ZipFile(file_path) as z:
        # Extract ZIP file contents in the same directory.
        z.extractall(os.path.split(file_path)[0])
    print("Unzipped", file_path)


if not os.path.isfile(ZIP):
    print(f"Downloading and extracting assests....")
    os.makedirs(DIR, exist_ok=True)
    urlretrieve(NET, TAR)
    untar(TAR)
    urlretrieve(URL, ZIP)
    unzip(ZIP)

# 1. Class file
classFile = path("coco_class_labels.txt")
with open(classFile) as fp:
    labels = fp.read().split("\n")
print(labels)

# 2. Model
modelFile = path("models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb")
configFile = path("models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    
# 3. Detect object
# For ach file in the directory
def detect_objects(net, im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects


FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


def display_text(im, text, x, y):
    text_size = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = text_size[0]
    baseline = text_size[1]
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, text, (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)


def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]
    print('Objects:', objects, objects.shape)

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        class_id = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        # Recover original cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        # Check if the detection is of good quality
        if score > threshold:
            display_text(im, "{}".format(labels[class_id]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10))
    plt.imshow(mp_img)
    plt.show()


im = cv2.imread(path("images/street.jpg"))
objects = detect_objects(net, im)
display_objects(im, objects)
