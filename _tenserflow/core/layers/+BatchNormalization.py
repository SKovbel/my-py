import numpy as np
import tensorflow as tf

input_array = np.random.randint(0, 256, (4, 8), dtype=np.uint8)

batch_normalization = tf.keras.layers.BatchNormalization()
batch_normalization2 = tf.keras.layers.Normalization(mean=2.0, variance=2.0)

normalized_array = batch_normalization(input_array, training=True)
normalized_array2 = batch_normalization2(input_array, training=True)

# Normiliza scaling data by input params
# BatchNormilize get data, calculate params and then scale

print("Original Array:")
print(input_array)

print("\nNormalized Array:")
print(normalized_array)

print("\nNormalized Array2:")
print(normalized_array2)