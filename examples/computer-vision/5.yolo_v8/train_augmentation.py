import os
import cv2
import albumentations as A
import argparse

from config import CLASSES

class DxTrainAugmentation:
    AUG_NUM = 10
    RULES = [
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-30, 30), p=0.5),
        #A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # A.Blur(blur_limit=3, p=0.2),
        # A.RandomCrop(width=256, height=256, p=0.4),
        A.Resize(width=640, height=640, p=1),
    ]

    def __init__(self, dir_image_in, dir_label_in, dir_image_out, dir_label_out):
        self.dir_image_in = dir_image_in
        self.dir_label_in = dir_label_in
        self.dir_image_out = dir_image_out
        self.dir_label_out = dir_label_out

        os.makedirs(self.dir_image_out, exist_ok=True)
        os.makedirs(self.dir_label_out, exist_ok=True)

    def process(self):
        i = 0
        for file_name in os.listdir(self.dir_image_in):
            i += 1
            label_name = f'{os.path.splitext(file_name)[0]}.txt'

            image_in = os.path.join(self.dir_image_in, file_name)
            label_in = os.path.join(self.dir_label_in, label_name)

            image_out = os.path.join(self.dir_image_out, file_name)
            label_out = os.path.join(self.dir_label_out, label_name)

            print(i, file_name) #, image_in, label_in, image_out, label_out)

            self.augmentate(image_in, label_in, image_out, label_out)

    def augmentate(self, image_in, label_in, image_out, label_out):
        image_in = cv2.imread(image_in)

        with open(label_in, 'r') as f:
            annotations = [list(map(float, line.strip().split())) for line in f.readlines()]

        bboxes = [obj[1:] for obj in annotations]
        classes = [CLASSES[0] for obj in annotations] # for now only 1 class copy to all masks
        transform = A.Compose(self.RULES, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        for i in range(self.AUG_NUM):
            augmented = transform(image=image_in, bboxes=bboxes, class_labels=classes)
            augmented_image = augmented['image']
            augmented_bboxes = augmented['bboxes']

            cv2.imwrite(image_out, augmented_image)
            with open(label_out, 'w') as f:
                for bbox in augmented_bboxes:
                    f.write(f'0 {bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video file.')
    parser.add_argument('image_in', type=str, help='Images in dir')
    parser.add_argument('label_in', type=str, help='Labels in dir')
    parser.add_argument('image_out', type=str, help='Images out dir')
    parser.add_argument('label_out', type=str, help='Labels out dir')
    args = parser.parse_args()

    augmentation = DxTrainAugmentation(args.image_in, args.label_in, args.image_out, args.label_out)
    augmentation.process()