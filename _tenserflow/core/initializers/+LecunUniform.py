import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import LecunUniform

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=LecunUniform())
layer = Dense(units=3, kernel_initializer='lecun_uniform')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.lecun_uniform)

output = layer(input)

print(layer.weights)
print(output.numpy())
