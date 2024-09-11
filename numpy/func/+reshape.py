import numpy as np

#input = input().strip().split(' ')
#result = np.asarray(input, dtype=np.int16)
#result = result.reshape(3, 3)
#print(result)


x = np.array([1,2,3,4,5,6,7,8,9,0])
print(x)
print(x.reshape(10, 1))


array = np.arange(2 * 3 * 4).reshape(2, 3, 4)
reshaped_array = array.reshape(-1, 4)
reshaped2_array = array.reshape(-1)
print(array)
print(reshaped_array)
print(reshaped2_array)  


a = [1,2,3,4,5]
for i in range(len(a)+1):
    print(a[:i])
for i in range(len(a)+1, 9):
    print(i)
