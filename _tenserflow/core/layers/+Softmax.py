import tensorflow as tf
from tensorflow.keras.layers import Softmax

# Your 1D tensor
input_tensor = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])

# Reshape the tensor to be 2D (batch_size, features)
input_tensor = tf.reshape(input_tensor, (1, -1))

# Apply Softmax layer along the last dimension
softmax_layer = Softmax(axis=-1)
softmax_output = softmax_layer(input_tensor)

# Print the result
print(softmax_output.numpy())
