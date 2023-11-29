import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeUniform

# Draws samples from a uniform distribution within [-limit, limit], where limit = sqrt(6 / fan_in)
#  (fan_in is the number of input units in the weight tensor).

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=HeUniform())
layer = Dense(units=3, kernel_initializer='he_uniform')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.he_uniform)

output = layer(input)

print(layer.weights)
print(output.numpy())
