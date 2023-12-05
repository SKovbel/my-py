XX = []
for i in range(1, 10):
    XX.append(5 * [i])
print(XX)
for X in XX:
    X[0] = 7
print(XX)
print(X)



A = [['apple'], ['banana'], ['cherry'], ['date']]
for index, value in enumerate(reversed(A)):
    value[0]=11
    print(f"Index: {index}, Value: {value}")
print("List:", A)