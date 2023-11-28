import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant

# Example input tensor
input_tensor = tf.constant([[1.0, 2.0, 3.0]])

# Define a Dense layer with constant initialization
constant_initializer = Constant(value=0.5)
dense_layer = Dense(units=3, kernel_initializer=constant_initializer)

# Apply the Dense layer with constant-initialized weights
output_tensor = dense_layer(input_tensor)

# Print the result
print(output_tensor.numpy())
