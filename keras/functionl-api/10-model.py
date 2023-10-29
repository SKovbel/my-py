import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image
tmp_dir = os.path.join(os.path.dirname(__file__), '../../tmp/keras-basic')

inputs = keras.Input(shape=(32,))
x = layers.Dense(64, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
mlp = keras.Model(inputs, outputs)

# vs

class MLP(keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.dense_1 = layers.Dense(64, activation='relu')
    self.dense_2 = layers.Dense(10)

  def call(self, inputs):
    x = self.dense_1(inputs)
    return self.dense_2(x)

mlp = MLP()
predict = mlp(tf.zeros((1, 32)))