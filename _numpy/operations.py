import numpy

a = numpy.asarray('1,2,3,4,5,6,7,8,9,0'.strip().split(','), dtype=numpy.float32)

print(numpy.flip(a))
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))
print(numpy.sum(a, axis=0))
print(numpy.prod(a))
print(numpy.min(a))
print(numpy.max(a))
