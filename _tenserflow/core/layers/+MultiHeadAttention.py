import tensorflow as tf
from tensorflow import keras

normal = keras.layers.LayerNormalization()
layer = keras.layers.MultiHeadAttention(
    num_heads=2,
    key_dim=2
)

x1 = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype="int32")

x2 = tf.constant([
    [100, 100, 100],
    [10000, 10000, 10000],
], dtype="int32")

x1 = tf.expand_dims(x1, axis=0)  # Add a batch dimension
x2 = tf.expand_dims(x2, axis=0)

x1 = normal(x1)
x2 = normal(x2)

y = layer(query=x1, value=x1, key=x1)

print(y)