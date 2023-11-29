import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal
# It draws samples from a truncated normal distribution centered on 0 
#   with stddev = sqrt(2 / fan_in) where fan_in is the number of input units in the weight tensor.


input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=HeNormal())
layer = Dense(units=3, kernel_initializer='he_normal')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.he_normal)

output = layer(input)

print(layer.weights)
print(output.numpy())
