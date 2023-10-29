import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(2, 2, strides=(1, 1), padding='valid', activation='relu', input_shape=(3, 3, 1), kernel_initializer=keras.initializers.Ones())
])

x = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

x = tf.reshape(x, [1, 3, 3, 1])
y = model.predict(x)

# [
#   [[1],[2],[3]]
#   [[4],[5],[6]]
#   [[7],[8],[9]]
# ]
print(x)

# [
#   [[12, 12], [16, 16]]
#   [[24, 24], [28, 28]]
# ]
print(y)
