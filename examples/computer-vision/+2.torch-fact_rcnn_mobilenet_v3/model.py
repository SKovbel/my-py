import os
from glob import glob

import torch
import models.model_fast_rcnn as fast_rcnn

from utils import path, DIR_MODEL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Model:
    # Create the model
    def create(self, do_eval=False):
        """
        Create PyTorch model
        """
        return fast_rcnn.create(do_eval)
        #return model_yolo.create(do_eval)

    # Save the model to the target dir
    def save(self, model: torch.nn.Module, epoch: int):
        """
        Saves a PyTorch model to a target directory.
        """
        save_path = path([DIR_MODEL, f"model_epoch_{epoch}"])
        if os.path.isfile(save_path):
            os.remove(save_path)
        torch.save(obj=model.state_dict(), f=save_path)


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
            model = self.create(do_eval=False)
            state = torch.load(check_point_path)
            model.load_state_dict(state)
            model.eval()
        elif empty_model:
            model = self.create()

        return model
