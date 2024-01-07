import os
import cv2
import numpy as np
import requests

path = lambda name: os.path.join(os.path.join(os.path.dirname(__file__), f"../../tmp/pytorch/cv1"), name)
sample_image = path("PennFudanPed/PNGImages/FudanPed00046.png")

# The Magic:
net = cv2.dnn.readNetFromONNX(path('onnx'))
image = cv2.imread(sample_image)
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (384, 384), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
biggest_pred_index = np.array(preds)[0].argmax()
print("Predicted class:", biggest_pred_index)


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
labels = {int(key): value for (key, value)
          in requests.get(LABELS_URL).json().items()}

print("The class", biggest_pred_index, "correspond to", labels[biggest_pred_index])