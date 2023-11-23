import tensorflow as tf
from tensorflow import keras
import numpy as np

#  (input - mean) / sqrt(var)
model = keras.Sequential([
    keras.layers.Normalization(mean=2.0, variance=2.0)
])

x = np.array([
    [0.0, 5.0, 10.0]
])
y = model.predict(x)

#[[-1.2  0.0  1.2]]
print(y)
