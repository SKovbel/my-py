# Import libraries
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from datetime import datetime

DIR = os.path.dirname(os.path.realpath(__file__))
path = lambda name: os.path.join(DIR, name)


# 1. Class file
labels=['object']

# 2. Model
modelFile = path("../data-yolo/result/weights/last.onnx")
print(modelFile)
net = cv2.dnn.readNetFromONNX(modelFile)

    
# 3. Detect object
# For ach file in the directory


FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
THRESHOLD = 0.25

def detect_objects(net, img, dim):
    blob = cv2.dnn.blobFromImage(img, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    result = net.forward()
    return result[0]


def display_objects(im, detections, threshold=THRESHOLD):
    H, W, _ = im.shape
    boxes = []
    classes = []
    confidences = []

    kx = W / 640
    ky = H / 640

    for detection in detections:
        confidence = detection[4]
        if confidence >= 0.5:
            class_ids = detection[5:]
            #if round(class_id) != 0:
            #    continue
            cx = int(detection[0] * kx)
            cy = int(detection[1] * ky)
            w = int(detection[2] * kx)
            h = int(detection[3] * ky)
            x = int(cx - 0.5 * w)
            y = int(cy - 0.5 * h)

            boxes.append([x, y, w, h])
            classes.append(class_ids)
            confidences.append(float(confidence))
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)

    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.26, 0.45)
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.imshow(mp_img)
    #plt.show()
    new_name = datetime.now().strftime("%d%m%Y_%H%M")
    cv2.imwrite(path('../tmp/r/' + new_name + '.png'), mp_img)


def display_text(im, text, x, y):
    text_size = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = text_size[0]
    baseline = text_size[1]
    cv2.rectangle(im, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(im, '{}'.format(text), (x, y - 5), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)

yolo_input_size = (640, 640)
img = cv2.imread(path("../data-yolo/samples/test/1.png"))
#img = cv2.resize(img, yolo_input_size)
objects = detect_objects(net, img, 640)
display_objects(img, objects)
