import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

x = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
y = np.array([x[i][0] ^ x[i][1] for i in range(len(x))])

def xLoss(x, y):
    loss = tf.keras.losses.binary_crossentropy(x, y)
    return loss

def xOptimizer():
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    return optimizer

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=xOptimizer(), loss=xLoss, metrics=['accuracy'])
model.fit(x, y, epochs=5000, verbose=0)

loss, accuracy = model.evaluate(x, y)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

predictions = model.predict(x)
print("Predictions:")
print(predictions)
