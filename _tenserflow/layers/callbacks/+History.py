import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# the History callback is automatically applied during training, and you can access the training history from the returned object after training. 
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

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

print(history.history)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.legend()
plt.show()
