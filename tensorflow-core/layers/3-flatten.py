import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Flatten()
])

x = np.array([[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]]
])
y = model.predict(x)

# [[ 1  2  3  4  5  6  7  8  9 10]]
print(y)
