import time

t = None
if t == 0:
    print('0')
if t is None:
    print('None')
if not t:
    print('Not')


a = [1,2,3,4,5]
b = [0,1,2,3,4,5,6,7,8,9]
for a1, b1 in zip(a, b):
    print(a1, b1)

def m():
    print('nn')
    return max((1,2))

values = [1,2,3,2,3,2,3,4,5,2]
ids = [i for i, v in enumerate(values) if v == m()]
print(ids)
