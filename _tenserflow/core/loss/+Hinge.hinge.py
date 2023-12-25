# The Hinge loss is commonly used in Support Vector Machines (SVM) for classification. 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Hinge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42)
# Convert the target labels to -1/+1 format for Hinge loss
y_hinge = 2 * y - 1
X_train, X_val, y_train, y_val = train_test_split(X, y_hinge, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(20,)))
model.add(Dense(1, activation='linear'))  # Output layer with linear activation for Hinge loss

# Compile the model with Hinge loss
model.compile(optimizer='adam', loss=Hinge(), metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
