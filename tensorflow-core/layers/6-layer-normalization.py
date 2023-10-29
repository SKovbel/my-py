import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.LayerNormalization()
])

x = np.array([
    [0.0, 5.0, 10.0]
])
y = model.predict(x)

#[[-1.2  0.0  1.2]]
print(y)
