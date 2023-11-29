import tensorflow as tf

# Create a PReLU layer
prelu_layer = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(0.25))

# Apply PReLU activation to some input data
input_data = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
output_data = prelu_layer(input_data)

print("Input Data:", input_data.numpy())
print("Output Data:", output_data.numpy())
