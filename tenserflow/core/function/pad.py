import tensorflow as tf

def causal_padding(inputs, kernel_size):
    pad_size = kernel_size - 1
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_size, 0], [0, 0]])
    return padded_inputs

input_data = tf.random.normal((1, 10, 1))  # (batch_size, timesteps, features)
print("Original input shape:", input_data.shape)

kernel_size = 3
padded_output = causal_padding(input_data, kernel_size)
print("Padded output shape:", padded_output.shape)

print("Original input data:\n", input_data.numpy())
print("Padded output data:\n", padded_output.numpy())