# The Kullback-Leibler (KL) Divergence loss is often used in probabilistic models to measure 
#   the difference between two probability distributions

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import KLDivergence
from sklearn.model_selection import train_test_split
import numpy as np

# Generate two synthetic probability distributions for demonstration
np.random.seed(42)
y_true = np.random.dirichlet(alpha=[1, 2, 3], size=100)  # True distribution
y_pred = np.random.dirichlet(alpha=[1.2, 2.5, 3.8], size=100)  # Predicted distribution

# Split the data into training and testing sets
y_true_train, y_true_test, y_pred_train, y_pred_test = train_test_split(y_true, y_pred, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(3, activation='softmax', input_shape=(3,)))

# Compile the model with KLDivergence loss
model.compile(optimizer='adam', loss=KLDivergence(), metrics=['categorical_crossentropy'])

# Train the model
model.fit(y_pred_train, y_true_train, epochs=50, verbose=1)

# Evaluate the model on the test set
kl_divergence, categorical_crossentropy = model.evaluate(y_pred_test, y_true_test, verbose=0)
print(f"Test KL Divergence: {kl_divergence}")
print(f"Test Categorical Crossentropy: {categorical_crossentropy}")
