import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import LecunNormal

input = tf.constant([[1.0, 2.0, 3.0]])

layer = Dense(units=3, kernel_initializer=LecunNormal())
layer = Dense(units=3, kernel_initializer='lecun_normal')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.lecun_normal)

output = layer(input)

print(layer.weights)
print(output.numpy())
