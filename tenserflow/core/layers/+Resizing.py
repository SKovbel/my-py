import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Resizing
import matplotlib.pyplot as plt

input_shape = (128, 128, 3) 
target_height = 64
target_width = 64

resize = Resizing(height=target_height, width=target_width, input_shape=input_shape)

image = np.random.randint(0, 256, input_shape, dtype=np.uint8)
image_normalized = image / 255.0
print(image_normalized[:1, :1, :5])
image_normalized = np.expand_dims(image_normalized, axis=0)
print(image_normalized[:1, :1, :5])
image_resized = resize(image_normalized)
print(image_resized[:1, :1, :1])
image_resized = np.squeeze(image_resized, axis=0)
print(image_resized[:1, :1, :1])

print(image.shape)
print(image_resized.shape)


plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Image')

plt.subplot(1, 2, 2)
plt.imshow(image_resized)
plt.title('Resized')

plt.show()