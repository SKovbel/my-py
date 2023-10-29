import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(5,))
])

x = np.array([[1, 2, 3, 4, 5]])
y = model.predict(x)

# [[1. 2. 3. 4. 5.]]
print(y)
