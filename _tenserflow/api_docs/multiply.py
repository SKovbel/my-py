import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf


print("TensorFlow version:", tf.__version__)

a = tf.constant((1, 2, 3, 4, 5, 6), shape=(2, 3), name='X_1')
b = tf.constant((7, 8, 9, 10, 11, 12), shape=(3, 2), name='X_2')

result = tf.linalg.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,
    name=None
)

print(result)
