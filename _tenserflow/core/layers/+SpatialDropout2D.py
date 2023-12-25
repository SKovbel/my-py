import tensorflow as tf
from tensorflow import keras
import numpy as np

layer = keras.layers.SpatialDropout2D(0.5)

x = np.array([
    [
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ]
    ]
])
y1 = layer(x)
y2 = layer(x, training=True)

# Inputs not set to 0 are scaled up by 1/(1 - rate)
print(y1) # [[1 2 3 4 5]]
print(y2) # [[2 4 0 8 0]]
