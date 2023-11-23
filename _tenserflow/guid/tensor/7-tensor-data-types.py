import tensorflow as tf
import numpy as np

print(tf.constant([2, 3, 4], dtype=tf.int16))
print(tf.constant([2.2, 3.3, 4.4], dtype=tf.float64))
print(tf.constant([2.2, 3.3, 4.4], dtype=tf.float64))
print(tf.constant([2.2, 3.3, 4.4], dtype=tf.float16))

# cast to an uint8 and lose the decimal precision
the_f16_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float16)
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)