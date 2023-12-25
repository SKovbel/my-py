import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

input_shape = (128, 128, 3) 
target_height = 64
target_width = 64

rescale = tf.keras.layers.Rescaling(scale=1/255)
resize = tf.keras.layers.Resizing(height=target_height, width=target_width, input_shape=input_shape)
dims = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0), input_shape=input_shape)
squeeze = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=0))

image = np.random.randint(0, 256, input_shape, dtype=np.uint8)

image_normalized = rescale(image)
image_normalized = dims(image_normalized)
image_resized = resize(image_normalized)
image_resized = squeeze(image_resized)

# another variants
if False:
    image_normalized = image / 255.0
    image_normalized = np.expand_dims(image_normalized, axis=0)
    image_normalized = rescale(image)
    image_resized = np.squeeze(image_resized, axis=0)


plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Image')

plt.subplot(1, 2, 2)
plt.imshow(image_resized)
plt.title('Resized')

plt.show()