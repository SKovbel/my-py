
A = [1,2,3,4,5,6,7]
B  ={'A': 1, 'B': 2, 'C':3, 'D':4, 'E':5, 'F':6, 'G': 7}

for item in A:
    print('A', item)

for i, item in enumerate(A):
    print('B', i, item)


for i in range(len(A)):
    print('C', i, A[i])

for item in B.values():
    print('D', item)

for key in B.keys():
    print('E', key, B.get(key))


for key, val in B.items():
    print('E', key, val)

print(sum(A))
print(max(A))

print(B.get('X', 'No'))
print(sum(B.values()))
print(max(B.values()))


key, value = B.popitem()
print(B)


ts = [11,2,3,4,5,6,7,8,-10, -1]
def f(ts):
    ts = [abs(val) for val in ts]
    return min(ts)
print(f(ts))
print(ts)

print(pow(2,2))