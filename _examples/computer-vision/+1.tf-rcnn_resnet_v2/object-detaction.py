import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras

DIR = os.getcwd()
path = lambda name: os.path.join(DIR, name)


LABELS = ['car', 'motocirle']
net = cv2.dnn.readNetFromONNX(path("model/onnx"))
# net = cv2.dnn.readNetFromTensorflow(path("model/tf/saved_model.pb"), path("model/tf/label_map.pbtxt"))
print(net.empty())


# 3. Detect object
# For ach file in the directory
def detect_objects(net, image, dim=224):
    image = cv2.resize(image, (224, 224))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    # image = image / 255.0  # Normalize pixel values to [0, 1]
    # image = np.expand_dims(image, axis=0)
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=False)
    blob = np.transpose(blob, (0, 2, 3, 1))  # Adjust the order of dimensions as needed
    net.setInput(blob)
    return net.forward()


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
            display_text(im, "{}".format(LABELS[class_id]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # Convert Image to RGB since we are using Matplotlib for displaying image
    mp_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(30, 10))
    plt.imshow(mp_img)
    plt.show()


image = cv2.imread(path("data/x/2.png"))
objects = detect_objects(net, image)
display_objects(image, objects)




# image = np.transpose(np.array(image), (2, 0, 1))
