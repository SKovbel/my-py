import tensorflow as tf
from tensorflow import keras
import numpy as np

layer = keras.layers.Flatten()

x = np.array([[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]]
])
y = layer(x)

print(y) # [[ 1  2  3  4  5  6  7  8  9 10]]
