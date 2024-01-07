# https://www.tensorflow.org/lite/models/modify/model_maker/object_detection
'''
sudo apt -y install libportaudio2
pip install -q --use-deprecated=legacy-resolver tflite-model-maker
pip install -q pycocotools
pip install -q opencv-python-headless==4.1.2.30
pip uninstall -y tensorflow && pip install -q tensorflow==2.8.0
'''

import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)