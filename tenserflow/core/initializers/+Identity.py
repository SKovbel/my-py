import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Identity

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=Identity(gain=1.0))
layer = Dense(units=3, kernel_initializer='identity')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.identity)

output = layer(input)

print(layer.weights)
print(output.numpy())
