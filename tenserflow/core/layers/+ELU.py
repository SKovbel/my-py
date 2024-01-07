import tensorflow as tf

# Create an ELU layer
elu_layer = tf.keras.layers.ELU(alpha=1.0)  # You can adjust the alpha parameter if needed

# Apply ELU activation to some input data
input_data = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
output_data = elu_layer(input_data)

print("Input Data:", input_data.numpy())
print("Output Data:", output_data.numpy())
