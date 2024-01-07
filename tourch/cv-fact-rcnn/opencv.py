import cv2
import onnx
import torch
import numpy as np
from model import Model
from data import path
import torchvision.models as models
from PIL import Image
from torchvision.transforms import functional as F
from data import FaceMaskDataset
from torch.utils.data import DataLoader

#################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
onnx_path = path('onnx.model')

dataset = FaceMaskDataset(what='test', transforms=None)
dataloader = DataLoader(dataset)
images, boxes, labels = next(iter(dataloader))
images = list(img.to(DEVICE) for img in images)

helper = Model()
model = helper.load(empty_model=True, epoch=1)
model.to(DEVICE)

# Create some sample input in the shape this model expects 
x = torch.randn(1, 3, 224, 224).to(DEVICE)
y = model(x)

'''
print('Model outputs: ', y[0]['boxes'].shape, y[0]['labels'].shape, y[0]['scores'].shape)
print('Model output boxes: ', y[0]['boxes'])
print('Model output labels: ', y[0]['labels'])
print('Model output scores: ', y[0]['scores'])
'''

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    x,
    onnx_path,
    opset_version=11,
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'},  'output': {0: 'batch_size'}})

'''onnx_model = onnx.load(onnx_path)
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')
'''

net = cv2.dnn.readNetFromONNX(onnx_path)