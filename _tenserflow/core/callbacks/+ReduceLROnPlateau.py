import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau

# The ReduceLROnPlateau callback in Keras is used to dynamically adjust the learning rate during training
#   based on the validation loss or a specified metric.
# It reduces the learning rate when a monitored quantity has stopped improving.
import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.randint(2, size=(100,))
X_val = np.random.rand(20, 10)
y_val = np.random.randint(2, size=(20,))

model = Sequential([
    Dense(units=32, activation='relu', input_shape=(10,)),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[reduce_lr])
