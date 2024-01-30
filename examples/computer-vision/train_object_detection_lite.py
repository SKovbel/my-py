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


spec = model_spec.get('efficientdet_lite0')
train_data, validation_data, test_data = object_detector.DataLoader.from_csv('gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv')
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
model.evaluate(test_data)
model.export(export_dir='.')
model.evaluate_tflite('model.tflite', test_data)
object_detector.DataLoader.from_pascal_voc(image_dir, annotations_dir, label_map={1: "person", 2: "notperson"})
spec = model_spec.get('efficientdet_lite0')
spec.config.var_freeze_expr = 'efficientnet'
#spec = model_spec.get('efficientdet_lite4')
model = object_detector.create(train_data, model_spec=spec, epochs=10, validation_data=validation_data)
model.export(export_dir='.')
model.export(export_dir='.', export_format=[ExportFormat.SAVED_MODEL, ExportFormat.LABEL])
config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='model_fp16.tflite', quantization_config=config)
