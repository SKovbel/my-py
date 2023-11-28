# The Huber loss is a combination of mean squared error and mean absolute error and is often used in regression problems


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Huber
from sklearn.model_selection import train_test_split
import numpy as np

# Generate a synthetic regression dataset for demonstration
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100, 1) * 2  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model
model = Sequential()
model.add(Dense(1, input_shape=(1,)))

# Compile the model with Huber loss
model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae', 'mse'])

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=1)

# Evaluate the model on the test set
loss, mae, mse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (Huber): {loss}")
print(f"Test Mean Absolute Error: {mae}")
print(f"Test Mean Squared Error: {mse}")
