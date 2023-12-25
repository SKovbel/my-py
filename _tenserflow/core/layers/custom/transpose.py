import tensorflow as tf

class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, permutation, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.permutation = permutation

    def call(self, inputs):
        return tf.transpose(inputs, perm=self.permutation)

# Example usage
input_tensor = tf.constant([[1, 2], [3, 4]])

# Create a TransposeLayer with a specific permutation
transpose_layer = TransposeLayer(permutation=[1, 0])

# Apply the layer to the input tensor
output_tensor = transpose_layer(input_tensor)

print("Input Tensor:")
print(input_tensor.numpy())
print("\nOutput Tensor:")
print(output_tensor.numpy())
