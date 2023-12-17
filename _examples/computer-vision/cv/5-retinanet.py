# https://keras.io/examples/vision/retinanet/
import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

DIR = os.path.join(os.getcwd(), f"../../tmp/cv-retinanet")
path = lambda name: os.path.join(DIR, name)
os.makedirs(DIR, exist_ok=True)

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = path("data.zip")
keras.utils.get_file(filename, url)


with zipfile.ZipFile(path("data.zip"), "r") as z_fp:
    z_fp.extractall(path("./"))
