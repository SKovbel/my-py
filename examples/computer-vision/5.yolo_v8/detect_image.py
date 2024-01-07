# Import libraries
import os
import cv2
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from datetime import datetime
from ultralytics import YOLO
import argparse

from config import CLASSES


class DxDetectImages:
    DIM = 640
    FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    THICKNESS = 1
    THRESHOLD = 0.25
    yolo_input_size = (640, 640)

    def __init__(self, model_path, images_in_dir, images_out_dir):
        self.images_in_dir = images_in_dir
        self.images_out_dir = images_out_dir
        self.model = YOLO(model_path)  # load a custom trained model

    def process(self):
        for file_name in os.listdir(self.images_in_dir):
            image_in = os.path.join(self.images_in_dir, file_name)
            image_out = os.path.join(self.images_out_dir, file_name)

            image = cv2.imread(image_in)
            result = self.detect_objects(image)
            self.display_objects(result, image, image_out)

    def detect_objects(self, image):
        return self.model(image)[0].boxes.data.tolist()

    def display_objects(self, result, image, image_out):
        boxes = []
        classes = []
        confidences = []

        H, W = image.shape[:2]
        kx, ky = W, H #/ 640

        for detection in result:
            confidence = detection[4]
            if confidence >= self.THRESHOLD:
                class_ids = detection[5:]
                #if round(class_id) != 0:
                #    continue
                cx = int(detection[0] * kx)
                cy = int(detection[1] * ky)
                w = int(detection[2] * kx)
                h = int(detection[3] * ky)
                x = int(cx - 0.5 * w)
                y = int(cy - 0.5 * h)
                x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])

                boxes.append([x, y, w, h])
                classes.append(class_ids)
                confidences.append(float(confidence))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.26, 0.45)
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_out, rgb_image)

    def display_text(self, image, text, x, y):
        text_size = cv2.getTextSize(text, self.FONTFACE, self.FONT_SCALE, self.THICKNESS)
        dim = text_size[0]
        baseline = text_size[1]
        cv2.rectangle(image, (x, y - dim[1] - baseline), (x + dim[0], y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, '{}'.format(text), (x, y - 5), self.FONTFACE, self.FONT_SCALE, (0, 255, 255), self.THICKNESS, cv2.LINE_AA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process multiple image files.')
    parser.add_argument('model_path', type=str, help='Model path')
    parser.add_argument('image_in', type=str, help='Images in dir')
    parser.add_argument('image_out', type=str, help='Images out dir')
    args = parser.parse_args()

    detector = DxDetectImages(args.model_path, args.image_in, args.image_out)
    detector.process()
