import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=Constant(value=0.5))

output = layer(input)
print(layer.weights)
print(output.numpy())
