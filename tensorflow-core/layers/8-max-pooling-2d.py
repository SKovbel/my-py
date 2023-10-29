import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')
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
#   [[5],[6]]
#   [[8],[9]]
# ]
print(y)
