import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Input data
X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
# Output data
y = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

def build_model(n_hidden_layers=5, hidden_units=3):
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=1, activation='relu'))  # First hidden layer
    
    for _ in range(n_hidden_layers - 1):
        model.add(Dense(hidden_units, activation='relu'))  # Additional hidden layers
    
    model.add(Dense(1))  # Output layer
    model.compile(optimizer='adam', loss='mse')
    return model

# Define the number of hidden layers and units
n_hidden_layers = 5
hidden_units = 3

# Build the model
model = build_model(n_hidden_layers=n_hidden_layers, hidden_units=hidden_units)

# Print the model summary
model.summary()

# Train the model
history = model.fit(X, y, epochs=200, verbose=0)

# Extract weights and biases
weights = model.get_weights()

# Predictions
predictions = model.predict(X)

# Plotting the input vs output
plt.scatter(X, y, color='blue', label='True values')
plt.plot(X, predictions, color='red', label='Predictions')

# Adding lines for the coefficients (weights and biases)
x_range = np.linspace(min(X), max(X), 100).reshape(-1, 1)

# Initialize the input for the first hidden layer
layer_output = x_range

# Iterate through each hidden layer
for i in range(n_hidden_layers):
    weight = weights[2 * i]  # Weight for the current hidden layer
    bias = weights[2 * i + 1]  # Bias for the current hidden layer
    layer_output = tf.nn.relu(np.dot(layer_output, weight) + bias)
    for j in range(hidden_units):
        plt.plot(x_range, layer_output[:, j], linestyle='dotted', label=f'Hidden layer {i+1}, node {j+1}')

# Final output layer
final_weight = weights[-2]
final_bias = weights[-1]
final_output = np.dot(layer_output, final_weight) + final_bias

plt.plot(x_range, final_output, linestyle='dashed', label='Final output')

# Labels and legend
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('NN Prediction and Weights/Biases Visualization')
plt.legend()
plt.show()
