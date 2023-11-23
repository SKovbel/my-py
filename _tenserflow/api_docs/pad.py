import tensorflow as tf

t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
paddings = tf.constant([[1, 1,], [2, 2]])

res = tf.pad(t, paddings, "CONSTANT")
print(res)

res = tf.pad(t, paddings, "REFLECT")
print(res)

res = tf.pad(t, paddings, "SYMMETRIC")
print(res)
