# https://keras.io/guides/keras_cv/classification_with_keras_cv/

import os
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv
from tensorflow.keras.optimizers import schedules
from keras import losses
from keras import optimizers
from keras import metrics

updir=os.path.join(os.path.dirname(__file__),  "../../../tmp/keras_cv")

# EfficientNetV2B0 is a great starting model when constructing an image classification pipeline. 
# another https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/backbones
classifier = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2_b0_imagenet_classifier"
)

classes = keras.utils.get_file(
    origin="https://gist.githubusercontent.com/LukeWood/62eebcd5c5c4a4d0e0b7845780f76d55/raw/fde63e5e4c09e2fa0a3436680f436bdcb8325aac/ImagenetClassnames.json",
    cache_dir='.', cache_subdir=updir
)

# Now that our classifier is built, let's apply it to this cute cat picture!
filepath = tf.keras.utils.get_file(origin="https://i.imgur.com/9i63gLN.jpg", cache_dir='.', cache_subdir=updir)
image = keras.utils.load_img(filepath)
image = np.array(image)
keras_cv.visualization.plot_image_gallery(
    np.array([image]), rows=1, cols=1, value_range=(0, 255), show=True, scale=4
)

# Next, let's get some predictions from our classifier:
predictions = classifier.predict(np.expand_dims(image, axis=0))
top_classes = predictions[0].argsort(axis=-1)
with open(classes, "rb") as f:
    classes = json.load(f)

top_two = [classes[str(i)] for i in top_classes[-2:]]
print("Top two classes are:", top_two)

