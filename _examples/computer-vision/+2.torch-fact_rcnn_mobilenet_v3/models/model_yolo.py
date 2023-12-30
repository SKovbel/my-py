import os
from glob import glob

import torch
from ultralytics import YOLO
from utils import path, DIR_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create the model
def create(do_eval=False):
    """
    Create PyTorch model
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, 3)
    model = model.to(DEVICE)
    if do_eval:
        model.eval()
    return model
