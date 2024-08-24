a = [1, 2, 3, 4, 5]

print(a[::])
print(a[2::])
print(a[:2:])
print(a[::2])
print(a[-1::])
print(a[:-1:])
print(a[::-1])



scores = {'a1': 1, 'a3': 123, 'a6': 23, 'a10': 2121, 'a11': 21}
minkey = min(scores, key=scores.get)
print(minkey)

def ll(x):
    print('_')
    return len(x)

a=[1, 2]
for i in range(1 - ll(a)):
    print('===', i)