import os
import cv2
import json
import albumentations as A

AUG_IMG_NUM = 10
DIR = os.path.dirname(os.path.realpath(__file__))
path = lambda name: os.path.join(DIR, name)

input_dir = path('../data-yolo/samples/train/objects/images')
input_label_dir = path('../data-yolo/samples/train/objects/labels')
output_dir = path('../data-yolo/samples/train-a/objects/images')
output_label_dir = path('../data-yolo/samples/train-a/objects/labels')
class_labels = ['object']

def augment_and_save(image_path, annotation_path, num_augmented_images):
    filename = os.path.basename(image_path)
    name = os.path.splitext(filename)[0]

    # Load image and annotations
    image = cv2.imread(image_path)
    with open(annotation_path, 'r') as f:
        annotations = [list(map(float, line.strip().split())) for line in f.readlines()]

    bboxes = [obj[1:] for obj in annotations]
    class_labels = ['object' for obj in annotations]
    # Define augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-30, 30), p=0.5),
        #A.Transpose(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        # A.Blur(blur_limit=3, p=0.2),
        A.RandomCrop(width=256, height=256, p=0.4),
        A.Resize(width=640, height=640, p=1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    for i in range(num_augmented_images):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        transformed_class_labels = augmented['class_labels']

        output_image_path = os.path.join(output_dir, f'augmented_{i}_{filename}')
        cv2.imwrite(output_image_path, augmented_image)

        output_annotation_path = os.path.join(output_label_dir, f'augmented_{i}_{name}.txt')
        with open(output_annotation_path, 'w') as f:
            for bbox in augmented_bboxes:
                f.write(f'0 {bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n')

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    i = 0
    for filename in os.listdir(input_dir):
        i += 1
        print(i, filename)
        image_path = os.path.join(input_dir, filename)
        annotation_path = os.path.join(input_label_dir, f"{os.path.splitext(filename)[0]}.txt")

        augment_and_save(image_path, annotation_path, AUG_IMG_NUM)
