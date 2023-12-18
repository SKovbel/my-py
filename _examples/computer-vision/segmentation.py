# https://www.tensorflow.org/tutorials/images/segmentation
import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt

DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../tmp/segmentation")
path = lambda name: os.path.join(DIR, name)

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir=DIR)
