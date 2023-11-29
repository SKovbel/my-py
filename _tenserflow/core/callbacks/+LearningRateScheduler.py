import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import LearningRateScheduler

# Generate some dummy data for illustration purposes
import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.randint(2, size=(100,))

# Define a simple model
model = Sequential([
    Dense(units=32, activation='relu', input_shape=(10,)),
    Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a step decay function for the learning rate
def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr

lr_scheduler = LearningRateScheduler(step_decay)

model.fit(X_train, y_train, epochs=30, batch_size=32, callbacks=[lr_scheduler])
