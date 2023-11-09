import numpy

input = input().strip().split(' ')
result = numpy.asarray(input, dtype=numpy.int16)
result = result.reshape(3, 3)
print(result)
