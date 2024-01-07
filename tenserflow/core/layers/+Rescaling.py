import tensorflow as tf
from tensorflow import keras
import numpy as np

# To rescale an input in the [0, 255] range to be in the [0, 1] range, you would pass scale=1./255.
# To rescale an input in the [0, 255] range to be in the [-1, 1] range, you would pass scale=1./127.5, offset=-1.

layer = keras.layers.Rescaling(scale=1/5, offset=-1)

x = np.array([
    [0.0, 5.0, 10.0]
])
y = layer(x)
print(y) # [[-1.0  0.0  1.0]]
