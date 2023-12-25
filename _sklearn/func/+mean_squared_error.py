from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,4,3,2,1]

# Calculate the mean squared error
mse = mean_squared_error(X, y)
print(f'Mean Squared Error: {mse}')

# Visualize the results (optional)
plt.scatter(X, y, color='black', label='Actual')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
