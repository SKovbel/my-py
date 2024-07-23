import numpy as np
import matplotlib.pyplot as plt

data = [10, 8, 10, 8, 8, 4, 8, 8, 8, 8]

mean = np.mean(data)
std_dev = np.std(data)
variance = np.var(data)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")

log_variance = np.log(variance)

def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = np.linspace(mean - 4*std_dev, mean + 4*std_dev, 1000)

y_std_dev = gaussian(x, mean, std_dev)
y_variance = gaussian(x, mean, np.sqrt(variance))
y_log_variance = gaussian(x, mean, np.log(std_dev))

plt.figure(figsize=(10, 6))
plt.plot(x, y_std_dev, label=f'Gaussian Distribution with $\sigma$={std_dev:.2f}', color='blue')
plt.plot(x, y_variance, label=f'Gaussian Distribution with $\sqrt{{\sigma^2}}$={np.sqrt(variance):.2f}', color='red', linestyle='--')
plt.plot(x, y_log_variance, label=f'Gaussian Distribution with $\log(\sigma)$={np.log(std_dev):.2f}', color='green', linestyle='-.')

plt.title('Gaussian Distributions for Dataset [10, 8, 10, 8, 8, 4]')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
