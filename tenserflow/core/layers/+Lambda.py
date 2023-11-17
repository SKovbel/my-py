import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

input_shape = (128, 128, 3) 

divide = tf.keras.layers.Lambda(lambda x: x/5, input_shape=input_shape)

image = np.random.randint(0, 256, input_shape, dtype=np.uint8)

image_div = divide(image)


plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Image')

plt.subplot(1, 2, 2)
plt.imshow(image_div)
plt.title('Resized')

plt.show()