import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
  y = x**2 + x

dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())


w = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='w')
b = tf.Variable([5.0, 6.0])
x = [[7., 8.]]

with tf.GradientTape(persistent=True) as tape:
  y = x @ w + b
  loss = tf.reduce_mean(y**2)
[dl_dw, dl_db] = tape.gradient(loss, [w, b])
dy_dw = tape.gradient(y, [w, b])
print(w.shape)
print(dl_dw.shape)
print('dl_dw', dl_dw.numpy())
print('dl_db', dl_db.numpy())
print('dy_dw', dy_dw)
print((x @ w + b).numpy())


my_vars = {
    'w': w,
    'b': b
}

grad = tape.gradient(loss, my_vars)
grad['b']
