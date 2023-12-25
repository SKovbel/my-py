import tensorflow as tf
from tensorflow import keras
import numpy as np

layer = keras.layers.InputLayer(input_shape=(5,))

x = [1, 2, 3, 4, 5]
y = layer(x)
print(y) # [1.t, 2.t, 3.t, 4.t, 5.t]


x = np.array([[1, 2, 3, 4, 5]])
y = layer(x)

# [[1.0, 2.0, 3.0, 4.0, 5.0]]
print(y)
