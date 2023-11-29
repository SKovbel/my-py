import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import BaseLogger

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

base_logger = BaseLogger(stateful_metrics=None)

model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val), callbacks=[base_logger])
