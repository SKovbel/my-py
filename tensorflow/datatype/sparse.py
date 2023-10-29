# https://www.tensorflow.org/guide/sparse_tensor
import numpy as np
import tensorflow as tf

def print_sparse_2d(sparce, msg = ''):
    printing = [['' for _ in range(sparce.dense_shape[1])] for _ in range(sparce.dense_shape[0])]
    for (index, value) in zip(sparce.indices, sparce.values):
        printing[index[0]][index[1]] = value.numpy()
    print(msg)
    print(np.matrix(printing))

# [0,0,0,10,0,0,0,0,0,0]
# [0,0,0,0,0,0,0,0,0,0]
# [0,0,0,0,20,0,0,0,0,0]
st1 = tf.sparse.SparseTensor(
    indices=[[0, 3], [2, 4]],
    values=[10, 20],
    dense_shape=[3, 10]
)
st2 = tf.sparse.from_dense([[1, 0, 0, 8], [0, 0, 0, 0], [0, 0, 3, 0]])
st3 = tf.sparse.to_dense(st2)

print('st1', st1)
print(print_sparse_2d(st1))

print('st2', st2)
print(print_sparse_2d(st2))

print('st3', st3)
print(st3.numpy()) # EagerTensor


print('\nManipulating sparse tensors')
st_a = tf.sparse.SparseTensor(
    indices=[[0, 2], [3, 4]],
    values=[31, 2], 
    dense_shape=[4, 10])

st_b = tf.sparse.SparseTensor(
    indices=[[0, 2], [0, 7]],
    values=[56, 38],
    dense_shape=[4, 10])

st_sum = tf.sparse.add(st_a, st_b)
print_sparse_2d(st_a, 'add1:')
print_sparse_2d(st_a, 'add2:')
print_sparse_2d(st_sum, 'Sum:')

st_c = tf.sparse.SparseTensor(indices=([0, 1], [1, 0], [1, 1]),
                       values=[13, 15, 17],
                       dense_shape=(2,2))

mb = tf.constant([[4], [6]])
product = tf.sparse.sparse_dense_matmul(st_c, mb)

print_sparse_2d(st_c, msg='Mul1:')
print(mb.numpy()) # EagerTensor
print(product.numpy()) # EagerTensor

print('Put sparse tensors together by using tf.sparse.concat and take them apart by using tf.sparse.slice.')
sparse_pattern_A = tf.sparse.SparseTensor(indices = [[2,4], [3,3], [3,4], [4,3], [4,4], [5,4]],
                         values = [1,1,1,1,1,1],
                         dense_shape = [8,5])
sparse_pattern_B = tf.sparse.SparseTensor(indices = [[0,2], [1,1], [1,3], [2,0], [2,4], [2,5], [3,5], 
                                              [4,5], [5,0], [5,4], [5,5], [6,1], [6,3], [7,2]],
                         values = [1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         dense_shape = [8,6])
sparse_pattern_C = tf.sparse.SparseTensor(indices = [[3,0], [4,0]],
                         values = [1,1],
                         dense_shape = [8,6])

sparse_patterns_list = [sparse_pattern_A, sparse_pattern_B, sparse_pattern_C]
sparse_pattern = tf.sparse.concat(axis=1, sp_inputs=sparse_patterns_list)
print(tf.sparse.to_dense(sparse_pattern))

sparse_slice_A = tf.sparse.slice(sparse_pattern_A, start = [0,0], size = [8,5])
sparse_slice_B = tf.sparse.slice(sparse_pattern_B, start = [0,5], size = [8,6])
sparse_slice_C = tf.sparse.slice(sparse_pattern_C, start = [0,10], size = [8,6])
print(tf.sparse.to_dense(sparse_slice_A))
print(tf.sparse.to_dense(sparse_slice_B))
print(tf.sparse.to_dense(sparse_slice_C))

st2_plus_5 = tf.sparse.map_values(tf.add, st2, 5)
print(tf.sparse.to_dense(st2_plus_5))   

st2_plus_5 = tf.sparse.SparseTensor(
    st2.indices,
    st2.values + 5,
    st2.dense_shape)
print(tf.sparse.to_dense(st2_plus_5))


# not finished
# next
# Using tf.sparse.SparseTensor with other TensorFlow APIs