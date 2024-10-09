import numpy as np
import keras
import random
import tensorflow as tf

random.seed(12)
np.random.seed(12)
tf.random.set_seed(12)

samples, timesteps, features = 2, 3, 3
inputs = np.array([
    [[1, 2, 3], [0, 0, 0], [0, 0, 0]],
    [[1, 2, 3], [1, 2, 3], [0, 0, 0]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
])
outputs = np.array([
    [1, 0, 0],
    [1, 2, 0],
    [1, 2, 3],
])

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(timesteps, features)))
model.add(keras.layers.LSTM(units=features, return_sequences=True))
model.add(keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(inputs, outputs)

outputs2 = model(inputs)

print(outputs2)
