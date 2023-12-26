import os
from glob import glob

import torch

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils import path, DIR_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelFastRCNN:
    def __init__(self):
        self.model = None

    # Create the model
    def create(self):
        """
        Create PyTorch model
        """
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, 3)
        model = model.to(DEVICE)
        return model


    # Save the model to the target dir
    def save(self, model: torch.nn.Module, epoch: int):
        """
        Saves a PyTorch model to a target directory.
        """
        target_dir_path = path(DIR_MODEL, True)
        check_point_name = f"model_epoch_{epoch}"
        model_save_path = target_dir_path / check_point_name
        torch.save(obj=model.state_dict(), f=model_save_path)


    # Load the model
    def load(self, empty_model=False, epoch=None):
        """
        Load a PyTorch model
        """
        if epoch:
            check_point_name = f"model_epoch_{epoch}"
            check_point_path = path([DIR_MODEL, check_point_name])
        else:
            save_path = path([DIR_MODEL, '*'])
            check_point_path = max(glob(save_path), key=os.path.getctime)

        model = None

        if os.path.isfile(check_point_path):
            model = self.create()
            state = torch.load(check_point_path)
            model.load_state_dict(state)
            model.eval()
        elif empty_model:
            model = self.create()
            model.eval()

        return model
