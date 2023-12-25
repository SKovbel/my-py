import tensorflow as tf
# https://www.tensorflow.org/tutorials/customization/basics

class NoBiasDense(tf.keras.layers.Layer):
    # where you can do all input-independent initialization
    def __init__(self, num_outputs):
        super(NoBiasDense, self).__init__()
        self.num_outputs = num_outputs

    # where you know the shapes of the input tensors and can do the rest of the initialization
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[int(input_shape[-1]), self.num_outputs])

    #  where you do the forward computation
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

inp = tf.ones([10, 5])

layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))
out = layer(inp)
print(out)
print(layer.variables)
print(layer.kernel, layer.bias)

layer = NoBiasDense(10)
out = layer(inp)
print(out)
print(layer.variables)
print(layer.kernel)