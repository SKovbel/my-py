import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[
    [[1.0], [2.0]],
    [[3.0], [4.0]]]], dtype=np.float32)

conv_transpose = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=2, strides=2, padding='valid')

# Manually set the weights and bias
conv_transpose.build(X.shape)
conv_transpose.set_weights([
    np.array([[
        [[1.0]], [[2.0]]],
        [[[2.0]], [[1.0]]]], dtype=np.float32),
    np.array([0.0], dtype=np.float32)])

Y = conv_transpose(X)

print("Input Tensor:")
print(X)
print("Output Tensor:")
print(Y.numpy())

def visualize_tensors(X, Y):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(X.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(Y.numpy().squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.show()

visualize_tensors(X, Y)