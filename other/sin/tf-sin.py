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

LR=0.01
EPOCHS = 1000

MUL = 20
MIN = -20
MAX = 20
MINT = 10
MAXT = 20

def func(x):
    return np.sin(x)

x = np.sort(np.random.uniform(2*MIN, 2*MAX, 2 * MUL * (MAX - MIN))).reshape((-1, 1))
y = func(x)

xt = np.random.uniform(MINT, MAXT, MUL * (MAXT - MINT)).reshape((-1, 1))
yt = func(xt)

model = models.Sequential(layers = [
    layers.InputLayer(1),
    layers.Dense(20, activation = 'tanh'),
    layers.Dense(1, activation = 'tanh')
])

optimizer = optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[])
result = model.fit(xt, yt, epochs=EPOCHS)

yp = model.predict(x)

plt.plot(x, y, c='b', label="sin")  # 'o' for circular points, 'b' for blue color
plt.scatter(xt, yt, marker='x', c='g', label="Train")  # 'o' for circular points, 'b' for blue color
plt.scatter(x, yp, s=6, c='r', label="Predicted")  # 'o' for circular points, 'b' for blue color
plt.legend(loc="lower left")

plt.show()
