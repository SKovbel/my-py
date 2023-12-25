import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.GlobalAveragePooling1D()
])

# remove timeseries dimension (as example)
x = tf.constant([
    [1, 4, 7],
    [12, 14, 16],
    [20, 30, 40]
])

x = tf.reshape(x, [1, 3, 3])
y = model.predict(x)

# [[
#   [1, 2, 3]
#   [12, 14, 16]
#   [20, 30, 40]
# ]]
print(x)

# [
#   [[11, 16, 21]]
# ]
print(y)
