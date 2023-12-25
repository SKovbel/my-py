import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Ones

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=Ones())
layer = Dense(units=3, kernel_initializer='ones')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.ones)

output = layer(input)

print(layer.weights)
print(output.numpy())
