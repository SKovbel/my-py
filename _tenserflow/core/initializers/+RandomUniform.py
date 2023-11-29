import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=None))
layer = Dense(units=3, kernel_initializer='random_uniform')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.random_uniform)

output = layer(input)

print(layer.weights)
print(output.numpy())
