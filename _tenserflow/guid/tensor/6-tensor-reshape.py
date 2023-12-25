import tensorflow as tf
import numpy as np

rank_0_tensor = tf.constant(4) # shape=()
rank_1_tensor = tf.constant([2.0, 3.0, 4.0]) # shape=(3,)
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16) # shape=(3,2)
rank_3_tensor = tf.constant([ # shape=(3, 2, 5)
  [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],])

print('rank_0_tensor =', rank_0_tensor.shape)
print('rank_1_tensor =', rank_1_tensor.shape)
print('rank_2_tensor =', rank_2_tensor.shape)
print('rank_3_tensor =', rank_3_tensor.shape)
print()
print(tf.reshape(rank_3_tensor, [-1]), "\n")
print(tf.reshape(rank_3_tensor, [6, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]), "\n")
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")
print(tf.reshape(rank_1_tensor, [3, 1]), "\n")
print(tf.reshape(rank_1_tensor, [3, 1, 1]), "\n")
print(tf.reshape(rank_1_tensor, [3, 1, 1, 1]), "\n")
print(tf.reshape(rank_1_tensor, [3, 1, 1, 1, 1, 1, 1]), "\n")

# Exception ragge
tf.reshape(rank_3_tensor, [7, -1])
