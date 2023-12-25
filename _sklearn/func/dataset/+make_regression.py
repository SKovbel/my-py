from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=33)

# Visualize the dataset
plt.scatter(X, y, color='black', label='Generated Data')
plt.title('Synthetic Regression Dataset')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.legend()
plt.show()
