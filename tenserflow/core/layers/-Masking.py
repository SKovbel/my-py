import numpy as np
import keras
import random
import tensorflow as tf

random.seed(12)
np.random.seed(12)
tf.random.set_seed(12)

samples, timesteps, features = 3, 4, 5
inputs = np.random.random([samples, timesteps, features]).astype(np.float32)
inputs[:, 2 , :] = 0.
inputs[:, 3, :] = 0.
inputs[1, 1, 1] = 0.1

model = keras.models.Sequential()
model.add(keras.layers.Masking(mask_value=0.))
model.add(keras.layers.LSTM(3, return_sequences=True))
output = model(inputs)

print(inputs)
print(output)

