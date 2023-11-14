import numpy as np
import timeit

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
N = 40000
M = 10

def using_tile():
    return np.tile(X, (N, 1))

def using_repeat():
    return np.repeat([X], N, axis=0)

# Measure time taken by each method
tile_time = timeit.timeit(using_tile, number=10)
repeat_time = timeit.timeit(using_repeat, number=10)

print(f"Time taken by numpy.tile(): {tile_time}")
print(f"Time taken by numpy.repeat(): {repeat_time}")
