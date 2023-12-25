# MultiOutputRegressor is a wrapper for turning a single-output estimator into a multi-output estimator.
# This can be useful when you have a single-output regression model, and you want to extend it to handle multiple outputs.


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# Generate a synthetic dataset with multiple outputs
X, y = make_regression(n_samples=100, n_features=10, n_targets=3, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a base regressor (you can use any single-output regressor)
base_regressor = RandomForestRegressor(n_estimators=10, random_state=42)

# Create a MultiOutputRegressor with the base regressor
multi_output_regressor = MultiOutputRegressor(base_regressor)

# Train the MultiOutputRegressor model
multi_output_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = multi_output_regressor.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(y_pred)
print(y_train)
# MultiOutputRegressor essentially trains a separate regressor for each target variable.
# You can replace RandomForestRegressor with any other single-output regression model depending on your specific requirements.