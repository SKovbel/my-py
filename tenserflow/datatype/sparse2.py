import tensorflow as tf
import numpy as np

dense_matrix = np.array([
    [1, 0, 0],
    [0, 0, 2],
    [0, 3, 0]
], dtype=np.int64)

indices = np.array(np.nonzero(dense_matrix)).T  # Find non-zero indices
values = dense_matrix[indices[:, 0], indices[:, 1]]  # Extract non-zero values
shape = dense_matrix.shape 

sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

# Convert sparse tensor back to a dense t
dense_again = tf.sparse.to_dense(sparse_tensor)

print("Dense Matrix", dense_matrix)
print("Indices", sparse_tensor.indices.numpy())
print("Values", sparse_tensor.values.numpy())
print("Dense Shape", sparse_tensor.dense_shape.numpy())

print("\nReconstructed Dense Matrix from Sparse Tensor:\n", dense_again.numpy())
