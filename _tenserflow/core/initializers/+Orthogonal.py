import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Orthogonal

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=Orthogonal(seed=None))
layer = Dense(units=3, kernel_initializer='orthogonal')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.orthogonal)

output = layer(input)

print(layer.weights)
print(output.numpy())
