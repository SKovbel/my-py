import tensorflow as tf
from tensorflow import keras
import numpy as np

# Layer
layer = keras.layers.Sigmoid(5, input_shape=(5,))

# Just layer
x = np.array([1, 2, 3, 4, 5])
y = layer(x)
print(y) # [[15. 15. 15. 15. 15.]]

# Model
model = keras.Sequential([
    layer
])
x = np.array([1, 2, 3, 4, 5])
y = model.predict(x)

print(y)
