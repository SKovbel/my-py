import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.GlobalMaxPooling1D()
])

# remove timeseries dimension (as example)
x = tf.constant([
    [1, 4, 7],
    [12, 14, 16],
    [20, 30, 40]
])

print('before', x)
x = tf.reshape(x, [1, 3, 3])
print('after', x)
y = model.predict(x)

# [[
#   [1, 2, 3]
#   [12, 14, 16]
#   [20, 30, 40]
# ]]
print(x)

# [
#   [[20, 30, 40]]
# ]
print(y)
