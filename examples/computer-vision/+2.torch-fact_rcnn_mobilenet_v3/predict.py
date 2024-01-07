import utils, predictions
from data import FaceMaskDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import predictions
from model import ModelFastRCNN

#################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

helper = ModelFastRCNN()
model = helper.load(empty_model=True, epoch=1)
#model = helper.create(True)

# Create DataLoaders from data.py
dataset = FaceMaskDataset(what='test', transforms=None)
dataloader = DataLoader(dataset)
images, boxes, labels = next(iter(dataloader))
images = list(img.to(DEVICE) for img in images)
predicts = model(images)
predicts = predictions.remove_low_risk_box(predictions=predicts, threshold=0.5)
predictions = predictions.apply_nms(predicts, 0.5)

## Display the predictions
utils.display_images(
    Output1=utils.display_boundary(images[0], predicts[0]['boxes'], predicts[0]['labels'], predicts[0]['scores']))
    #Output2=utils.display_boundary(images[1], predicts[1]['boxes'], predicts[1]['labels'], predicts[1]['scores']))
