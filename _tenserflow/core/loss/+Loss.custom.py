import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss
from sklearn.model_selection import train_test_split
import numpy as np

# Custom Mean Absolute Error (MAE) class
class CustomMAE(Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

# Generate a synthetic dataset for demonstration
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 2 * X + 1 + np.random.randn(100, 1) * 2  # Linear relationship with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple linear regression model
model = Sequential()
model.add(Dense(1, input_shape=(1,)))

# Create an instance of the custom MAE class
custom_mae = CustomMAE()

# Compile the model with the custom MAE as the loss function
model.compile(optimizer='adam', loss=custom_mae)

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=1)

# Evaluate the model on the test set
mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on Test Set: {mae}")
