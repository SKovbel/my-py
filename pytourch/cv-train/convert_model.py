import os
import sys
import torch
import cv_model

sys.path.append(os.path.join(os.path.dirname(__file__), "cv_utils"))

path = lambda name: os.path.join(os.path.join(os.path.dirname(__file__), f"../../tmp/pytorch/cv1"), name)
os.makedirs(path(''), exist_ok=True)
model = cv_model.get_model_instance_segmentation(2)
model.load_state_dict(torch.load(path('model')))
model.eval()


dummy_input = torch.randn(1, 3, 384, 384)
torch.onnx.export(model, dummy_input, path("onnx"), verbose=True)
