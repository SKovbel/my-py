import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LambdaCallback

# Generate some dummy data for illustration purposes
import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.randint(2, size=(100,))
X_val = np.random.rand(20, 10)
y_val = np.random.randint(2, size=(20,))

# Define a simple model
model = Sequential([
    Dense(units=32, activation='relu', input_shape=(10,)),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a custom function to be executed at the beginning of each epoch
def on_epoch_begin(epoch, logs):
    print(f"\nStarting epoch {epoch + 1}")

# Define a custom function to be executed at the end of each epoch
def on_epoch_end(epoch, logs):
    print(f"\nEnd of epoch {epoch + 1}. Training accuracy: {logs['accuracy']}, Validation accuracy: {logs['val_accuracy']}")

# Define a custom function to be executed during batch iteration
def on_batch_end(batch, logs):
    print(f"\nEnd of batch {batch + 1}. Training loss: {logs['loss']}")

# Create a LambdaCallback with custom functions
lambda_callback = LambdaCallback(
    on_epoch_begin=lambda epoch, logs: on_epoch_begin(epoch, logs),
    on_epoch_end=lambda epoch, logs: on_epoch_end(epoch, logs),
    on_batch_end=lambda batch, logs: on_batch_end(batch, logs)
)

# Train the model with the LambdaCallback
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val), callbacks=[lambda_callback])
