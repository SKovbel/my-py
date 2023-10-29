import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dropout(0.5)
])

x = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0]
])
y1 = model.predict(x)
y2 = model.layers[0](x, training=True)

# Inputs not set to 0 are scaled up by 1/(1 - rate)
# [[1 2 3 4 5]]
print(y1)
# [[2 4 0 8 0]]
print(y2)
