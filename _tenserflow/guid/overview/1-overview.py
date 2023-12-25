# https://www.tensorflow.org/api_docs/python/tf/Tensor
# https://www.tensorflow.org/guide/basics
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print('x =', x)
print('shape =', x.shape)
print('dtype =', x.dtype)

print('x + x =', x + x)
print('5 * x =', 5 * x)
print('x @ tf.transpose(x) =', x @ tf.transpose(x))
print('tf.concat([x, x, x], axis=0) =', tf.concat([x, x, x], axis=0))
print('tf.concat([x, x, x], axis=1) =', tf.concat([x, x, x], axis=1))
print('tf.nn.softmax(x, axis=-1) =', tf.nn.softmax(x, axis=-1))
print('tf.reduce_sum(x) =', tf.reduce_sum(x))



c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
e = tf.matmul(c, d)

print('Variables')
var = tf.Variable([0.0, 0.0, 0.0])
var.assign([1, 2, 3])
var.assign_add([1, 1, 1])
print(var)

x = tf.Variable(1.0)
def f(x):
    y = x**2 + 2*x - 5
    return y
print(f(x))

with tf.GradientTape() as tape:
    y = f(x)
g_x = tape.gradient(y, x)  # g(x) = dy/dx
print(g_x)

print('Automatic differentiation')
@tf.function
def my_func(x):
    print('Tracing.\n')
    return tf.reduce_sum(x)

x = tf.constant([1, 2, 3])
print(my_func(x))
x = tf.constant([10, 9, 8])
print(my_func(x))
x = tf.constant([10.0, 9.1, 8.2], dtype=tf.float32)
print(my_func(x))
