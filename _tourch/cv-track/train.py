"""
https://github.com/Followb1ind1y/Face-Mask-Detection
pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
"""

import utils, engine, predictions
from data import FaceMaskDataset, path

import numpy as np
import albumentations as A

import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model import ModelFastRCNN

# Set the manual seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Setup parameters
SET = ['train', 'val', 'test']
BATCH = 4
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create data augmentation
augmentations = [
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),]
data_transform = A.Compose(augmentations, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Create DataLoaders from data.py
datasets = {x: FaceMaskDataset(what=x, transforms=data_transform) for x in SET}
dataloaders = {x: DataLoader(datasets[x], batch_size=BATCH, shuffle=True, collate_fn=datasets[x].collate_fn) for x in SET}
dataset_sizes = {x: len(datasets[x]) for x in SET}

#  Create Object Detection Model
weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(features, 3)
model = model.to(DEVICE)

## Model inItialization
params = [p for p in model.parameters() if p.requires_grad]
optimizer_RCNN = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
exp_lr_scheduler_RCNN = torch.optim.lr_scheduler.StepLR(optimizer_RCNN, step_size=7, gamma=0.1)

# Trainer
trainer = engine.Trainer(model=model,
                         dataloaders=dataloaders,
                         epochs=EPOCHS,
                         metric=None,
                         criterion=None,
                         optimizer=optimizer_RCNN,
                         scheduler=exp_lr_scheduler_RCNN,
                         save_dir=path(["models", "RCNN_Model_Output"], True),
                         device=DEVICE)

## Training process
model_results = trainer.train_model()

## Evaluate the model
images, boxes, labels = next(iter(dataloaders['test']))
images = list(img.to(DEVICE) for img in images)

model.eval()
predictions = model(images)
predictions = predictions.remove_low_risk_box(predictions=predictions, threshold=0.5)
predictions = predictions.apply_nms(predictions, 0.5)

## Display the predictions
utils.display_images(
    Output1=utils.display_boundary(images[0], predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']),
    Output2=utils.display_boundary(images[1], predictions[1]['boxes'], predictions[1]['labels'], predictions[1]['scores']))
