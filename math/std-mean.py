import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return -x**2 + 2*x + 1

x = np.linspace(-2, 4, 400)
y = func(x)

mean_y = np.mean(y)
std_y = np.std(y)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='$y = -x^2 + 2x + 1$', color='blue')

plt.axhline(mean_y, color='green', linestyle='--', label=f'Mean: {mean_y:.2f}')

plt.axhline(mean_y + std_y, color='red', linestyle='--', label=f'Mean + Std: {mean_y + std_y:.2f}')
plt.axhline(mean_y - std_y, color='red', linestyle='--', label=f'Mean - Std: {mean_y - std_y:.2f}')

plt.title('Function $y = -x^2 + 2x + 1$ with Mean and Standard Deviation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.show()