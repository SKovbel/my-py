import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import VarianceScaling

input = tf.constant([[1.0, 2.0, 3.0]])
# Initializer that adapts its scale to the shape of its input tensors.
layer = Dense(units=3, kernel_initializer=VarianceScaling(scale=1.0, mode='fan_in', 
                                                          distribution='truncated_normal', seed=None))
layer = Dense(units=3, kernel_initializer='variance_scaling')
layer = Dense(units=3, kernel_initializer=tf.keras.initializers.variance_scaling)

output = layer(input)

print(layer.weights)
print(output.numpy())
