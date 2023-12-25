import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3, random_state=42)
y_one_hot = to_categorical(y, num_classes=3)
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(20,)))
model.add(Dense(3, activation='softmax'))  # Output layer with softmax for multi-class classification

model.compile(optimizer='adam', loss=CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")
