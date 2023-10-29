import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([
    keras.layers.Dense(5, input_shape=(5,), activation='linear', kernel_initializer=keras.initializers.Ones())
])

x = np.array([
    [1, 2, 3, 4, 5]
])
y = model.predict(x)

# [[15. 15. 15. 15. 15.]]
print(y)
