import random
import matplotlib.pyplot as plt

x = [i for i in range(-10, 10)]
y = [i**2 for i in x]
y2 = [random.randint(0, 10) for i in x]

print(x)
print(y)
print(y2)
#plt.plot([9,4,1,0,1,4,9])
#plt.ylabel('Y')
#plt.show()

plt.plot(x, y)
plt.ylabel('Y')
plt.show()

bins = 10
plt.hist(y2, bins, edgecolor='black', facecolor='blue')
plt.ylabel('Y')
plt.show()
