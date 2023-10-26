import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

MIN=-20
MAX=20
SAMPLES=1000
DIFF = 0.01
LR=0.1
EPOCHS = 100

def func(x):
    return np.sin(x)

x = np.random.uniform(MIN, MAX, SAMPLES).reshape((-1, 1))
y = func(x)

xt = np.random.uniform(MIN, MAX, SAMPLES).reshape((-1, 1))
yt = func(xt)

model = models.Sequential(layers = [
    layers.InputLayer(1),
    layers.Dense(20, activation = 'tanh'),
    layers.Dense(1, activation = 'tanh')
])

optimizer = optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[])
result = model.fit(x, y, epochs=EPOCHS)

yp = model.predict(xt)

diff = np.abs(yp - yt) / yt
true_cnt = sum(np.where(diff < DIFF, 1, 0))
false_cnt = len(diff) - true_cnt

plt.subplot(1, 2, 1)
plt.scatter(xt, yp, marker='o', color='b')  # 'o' for circular points, 'b' for blue color
plt.title('Predicted')

plt.subplot(1, 2, 2)
plt.scatter(xt, yt, marker='o', color='b')  # 'o' for circular points, 'b' for blue color
plt.title('Train')

plt.show()
