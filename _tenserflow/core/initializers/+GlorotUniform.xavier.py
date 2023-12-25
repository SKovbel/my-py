import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=GlorotUniform())
layer = Dense(units=3, kernel_initializer='glorot_uniform')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.glorot_uniform)

output = layer(input)

print(layer.weights)
print(output.numpy())
