# https://www.tensorflow.org/guide/data#basic_mechanics
import tensorflow as tf

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.set_printoptions(precision=4)

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

# Basic mechanics
print('Basic mechanics')
print(dataset)
for elem in dataset:
  print(elem.numpy())

it = iter(dataset)
print('iter')
print(next(it).numpy())
print(next(it).numpy())

print('reduce')
print(dataset.reduce(0, lambda state, value: state + value).numpy())
dataset.map(lambda value: print(value) or 1)


# Dataset structure
print('\nDataset structure')
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform((4, 10)))
print('1:', dataset1.element_spec)

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print('2:', dataset2.element_spec)

dataset3 = tf.data.Dataset.zip((dataset, dataset1, dataset2))
print('3:', dataset3.element_spec)

# Dataset containing a sparse tensor.
dataset4 = tf.data.Dataset.from_tensors(tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4]))
print('4:', dataset4.element_spec)


print(dataset4.element_spec.value_type)


dataset1 = tf.data.Dataset.from_tensor_slices(
    tf.random.uniform([4, 10], minval=1, maxval=10, dtype=tf.int32))
print('d1:', dataset1)
for z in dataset1:
  print(z.numpy())


dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print('d2:', dataset2)

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print('d3:', dataset3)
for a, (b,c) in dataset3:
    print('shapes: {a.shape}, {b.shape}, {c.shape}'.format(a=a, b=b, c=c))


# next Reading input data