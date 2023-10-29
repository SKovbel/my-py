import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from PIL import Image
tmp_dir = os.path.join(os.path.dirname(__file__), '../../tmp/keras-basic')


units = 32
timesteps = 10
input_dim = 5
batch_size = 16


@keras.saving.register_keras_serializable()
class CustomRNN(layers.Layer):
    def __init__(self):
        super().__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        return self.classifier(features)


# Note that you specify a static batch size for the inputs with the `batch_shape`
# arg, because the inner computation of `CustomRNN` requires a static batch size
# (when you create the `state` zeros tensor).
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)

model = keras.Model(inputs, outputs)

rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, 10, 5)))

rnn_model = CustomRNN()

# tf.zeros((1, 2, 3))
# [
#   [
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0]
#   ]
# ]

