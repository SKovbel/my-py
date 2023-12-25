import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Zeros

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=Zeros())
layer = Dense(units=3, kernel_initializer='zeros')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.zeros)

output = layer(input)

print(layer.weights)
print(output.numpy())
