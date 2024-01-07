import tensorflow as tf
import numpy as np

rank_0_tensor = tf.constant(4) # shape=()
rank_1_tensor = tf.constant([2.0, 3.0, 4.0]) # shape=(3,)
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16) # shape=(3,2)
rank_3_tensor = tf.constant([ # shape=(3, 2, 5)
  [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],])

# Broadcasting is a concept borrowed from the equivalent feature in NumPy.
# In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.

x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)

print("3x1 * 1x4 = 3x4")
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))

print("Strench the same as before * ")
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])
print(x_stretch * y_stretch)  # Again, operator overloading

print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))
