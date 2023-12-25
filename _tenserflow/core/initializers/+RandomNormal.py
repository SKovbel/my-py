import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None))
layer = Dense(units=3, kernel_initializer='random_normal')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.random_normal)

output = layer(input)

print(layer.weights)
print(output.numpy())
