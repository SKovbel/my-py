from sklearn.datasets import make_regression
from sklearn.multioutput import RegressorChain
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset with multiple outputs
X, y = make_regression(n_samples=100, n_features=10, n_targets=3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a base regressor (you can use any regressor)
base_regressor = RandomForestRegressor(n_estimators=10, random_state=42)

# Create a RegressorChain with the base regressor
regressor_chain = RegressorChain(base_regressor, order=[0, 1, 2])  # Define the order of chaining

# Train the RegressorChain model
regressor_chain.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor_chain.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# Remember that the order parameter in RegressorChain determines the order in which the targets are chained.
# You can adjust it based on your specific problem. The example order [0, 1, 2] means that the first target is predicted directly,
# and then its prediction is used as a feature for predicting the second target, and so on.
