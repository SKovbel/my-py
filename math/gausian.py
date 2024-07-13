import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Given dataset
data = np.array([10.0, 8.0, 10.0, 8.0, 8.0, 4.0])

# Calculating mean, variance, and standard deviation
mean = np.mean(data)
variance = np.var(data, ddof=1)
std_dev = np.std(data, ddof=1)

print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")

#
# Mean - avarage value of all points
# 
# Standard deviation - avarage distance between points and mean
#

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
#y = stats.norm.pdf(x, mean, std_dev)
y = stats.norm.pdf(x, mean, std_dev)
y = gaussian(x, mean, std_dev)

plt.figure(figsize=(12, 12))
plt.plot(x, y, label='Gaussian Distribution')
plt.axvline(mean, color='r', linestyle='--', label=f'Mean = {mean:.2f}')

plt.axvline(mean - std_dev, color='g', linestyle='--', label=f'-1 Std Dev = {std_dev:.2f}')
plt.axvline(mean + std_dev, color='g', linestyle='--', label=f'+1 Std Dev = {std_dev:.2f}')
plt.axvline(mean - 2*std_dev, color='b', linestyle='--', label='-2 Std Dev')
plt.axvline(mean + 2*std_dev, color='b', linestyle='--', label='+2 Std Dev')
plt.axvline(mean - 3*std_dev, color='y', linestyle='--', label='-3 Std Dev')
plt.axvline(mean + 3*std_dev, color='y', linestyle='--', label='+3 Std Dev')

plt.text(mean + 0.1*std_dev, 0.07, f'Variance: {variance:.2f}', fontsize=12, color='orange')

plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
