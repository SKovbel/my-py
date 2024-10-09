import os
import numpy as np
import splitfolders
import xml.etree.ElementTree as ET
from PIL import Image

import torch
import torchvision

DIR_DATA = os.path.join(os.path.dirname(__file__), "../../tmp/torch-cv")
DIR_IN = 'in'
DIR_OUT = 'out'
DIR_SPLIT = 'split'
DIR_MODEL = 'model'

def path(name, create=False):
    dir = os.path.join(DIR_DATA, *name) if type(name) is list else os.path.join(DIR_DATA, name)
    if create:
        os.makedirs(dir, exist_ok=True)
    return dir

def split(in_dir, out_dir):
    splitfolders.ratio(in_dir, output=out_dir, seed=42, ratio=(0.8, 0.1, 0.1))

def get_annotations_boxes_from_xml(dir):
    tree = ET.parse(dir)
    root = tree.getroot()

    annotations, labels = [], []

    for neighbor in root.iter('bndbox'):
        xmin = int(neighbor.find('xmin').text)
        ymin = int(neighbor.find('ymin').text)
        xmax = int(neighbor.find('xmax').text)
        ymax = int(neighbor.find('ymax').text)

        annotations.append([xmin, ymin, xmax, ymax])

    for neighbor in root.iter('object'):
        label = neighbor.find('name').text
        if label == 'without_mask':
            labels.append(2)
        else:
            labels.append(1)

    return annotations, labels


class FaceMaskDataset(torch.utils.data.Dataset):
    def __init__(self, what, transforms=None):
        self.root = path([DIR_SPLIT, what], True)
        self.imgs = list(sorted(os.listdir(path([self.root, 'images']))))
        self.anns = list(sorted(os.listdir(path([self.root, 'annotations']))))
        self.img_dir = path([self.root, 'images'])
        self.ann_dir = path([self.root, 'annotations'])

        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        curr_img_dir = os.path.join(self.img_dir, self.imgs[idx])
        curr_ann_dir = os.path.join(self.ann_dir, self.anns[idx])

        image = Image.open(curr_img_dir, mode='r').convert('RGB')
        boxes, labels = get_annotations_boxes_from_xml(curr_ann_dir)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        if self.transforms:
            pass
            #image2 = self.transforms(image=np.array(image), bboxes=boxes, category_ids=labels)

        tenn = torchvision.transforms.ToTensor()
        image = tenn(image)
        return image, boxes, labels

    def collate_fn(self, batch):
        return tuple(zip(*batch))


# Test
if __name__ == "__main__":
    split(path(DIR_IN), path(DIR_SPLIT, True))

    for what in ['train', 'val', 'test']:
        image_datasets = FaceMaskDataset(what, transforms=None)
        print(image_datasets[1])
