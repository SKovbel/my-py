# The TerminateOnNaN callback in Keras is used to terminate training if a NaN (Not a Number) or Inf (Infinity) value is encountered during training. 
# This can be useful to prevent the model from continuing training when the loss becomes unstable or explodes

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TerminateOnNaN
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

terminate_on_nan = TerminateOnNaN()

model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), callbacks=[terminate_on_nan])
