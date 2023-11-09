import numpy



arr =[[1,2], [3,4], [5,6]]
a = numpy.asarray(arr, dtype=numpy.int16)
print('matrix:', a)
result = a.transpose()
print('transpose:', result)
result = a.flatten()
print('flatten:', result)
result = numpy.concatenate((a, a))
print('concatenate:', result)


print(numpy.sum(a, axis=0))
print(numpy.sum(a, axis=1))
print(numpy.max(a, axis=1))
print(numpy.min(a, axis=1))
print(numpy.mean(a, axis=1))
print(numpy.var(a, axis=0))
print(numpy.std(a))
