import numpy as np
import tensorflow as tf
# https://www.tensorflow.org/tutorials/customization/custom_layers

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
print(np.array([element.numpy() for element in ds_tensors]))

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
print(np.array([element.numpy() for element in ds_file]))

ds_tensors = ds_tensors.map(tf.math.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
print(np.array([element.numpy() for element in ds_tensors]))
print(np.array([element.numpy() for element in ds_file]))

# iteration
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)