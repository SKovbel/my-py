import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ProgbarLogger

import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.randint(2, size=(100,))

model = Sequential([
    Dense(units=32, activation='relu', input_shape=(10,)),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
progbar_logger = ProgbarLogger()

model.fit(X_train, y_train, epochs=3, batch_size=32)
