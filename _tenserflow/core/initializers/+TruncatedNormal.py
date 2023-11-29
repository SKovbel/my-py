import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05, seed=None))
layer = Dense(units=3, kernel_initializer='truncated_normal')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.truncated_normal)

output = layer(input)

print(layer.weights)
print(output.numpy())
