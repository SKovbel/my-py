import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 1. Create the Dataset
# Input data
X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
# Output data
y = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# 2. Define the Neural Network
def build_model(n, m):
    model = Sequential()
    model.add(Dense(n, input_dim=1, activation='relu'))  # First dense layer with n neurons
    model.add(Dense(m, activation='relu'))  # Second dense layer with m neurons
    model.add(Dense(1))  # Output layer with 1 neuron
    model.compile(optimizer='adam', loss='mse')
    return model

# Define the number of neurons in each hidden layer
n_neurons = 5
m_neurons = 3

# Build the model
model = build_model(n_neurons, m_neurons)

# Print the model summary
model.summary()

# 3. Train the Neural Network
history = model.fit(X, y, epochs=200, verbose=0)

# 4. Extract Weights and Biases
weights = model.get_weights()

# 5. Calculate Overall Weights and Biases
def calculate_overall_weights_biases(weights, n_neurons, m_neurons):
    W1, b1 = weights[0].reshape((1, n_neurons)), weights[1]
    W2, b2 = weights[2].reshape((n_neurons, m_neurons)), weights[3]
    W3, b3 = weights[4].reshape((m_neurons, 1)), weights[5]

    # Calculate intermediate outputs
    intermediate_W1_W2 = np.dot(W1, W2)
    intermediate_b1_W2 = np.dot(b1, W2) + b2

    # Calculate overall weights and biases
    overall_weight = np.dot(intermediate_W1_W2, W3)
    overall_bias = np.dot(intermediate_b1_W2, W3) + b3

    return overall_weight, overall_bias

# Calculate overall weights and biases
overall_weight, overall_bias = calculate_overall_weights_biases(weights, n_neurons, m_neurons)

print("Overall Weight:", overall_weight)
print("Overall Bias:", overall_bias)

# 6. Plot True Values and Predictions
# Predictions
predictions = model.predict(X)

# Compute the line using overall weights and biases
x_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)
overall_line = overall_weight * x_range + overall_bias

# Plotting the input vs output
plt.scatter(X, y, color='blue', label='True values')
plt.plot(X, predictions, color='red', label='NN Predictions')
plt.plot(x_range, overall_line, color='green', linestyle='dashed', label='Overall Linear Approximation')

# Labels and legend
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('NN Prediction and Overall Linear Approximation')
plt.legend()
plt.show()
