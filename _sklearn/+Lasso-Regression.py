# Lasso regression is a linear regression technique that includes an L1 regularization term in the cost function.
# This helps in feature selection by pushing the coefficients of less important features to zero.
# In scikit-learn, you can use the Lasso class to perform Lasso regression.
# You can adjust the alpha parameter based on the amount of regularization you want to apply.
# Higher values of alpha increase the regularization strength, and the model tends to shrink coefficients more aggressively.

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate a synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Lasso regression model
lasso_model = Lasso(alpha=1.0)  # Alpha is the regularization strength

# Train the model
lasso_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lasso_model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results (optional)
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

