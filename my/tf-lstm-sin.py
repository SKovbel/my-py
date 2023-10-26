import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import models

MIN=-20
MAX=20
SAMPLES=1000
DIFF = 0.1

EPOCHS = 100
TIME_STEP = 2

def func(x):
    return x
    #return np.sin(x)

x = np.random.uniform(MIN, MAX, SAMPLES)
y = func(x)

x = x.reshape(len(x) // 20, 20, 1)
y = y.reshape(len(y) // 20, 20, 1)


# test x,y
xt = np.random.uniform(MIN * MIN, MAX * MAX, SAMPLES)
yt = func(xt)

xt = xt.reshape(len(xt) // 20, 20, 1)
yt = yt.reshape(len(yt) // 20, 20, 1)

LSTM_NODES = 10
model = models.Sequential(layers = [
    layers.LSTM(LSTM_NODES, activation = 'tanh'),
    layers.Dense(1, activation = 'tanh')
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=EPOCHS)

for i in range(len(xt)):
    yp = model.predict(xt[i])
    cmp = np.abs(yp - yt) / yt <= DIFF
    print(cmp)
    true_cnt = sum(1 for j in range(len(cmp)) if cmp[i][j] == True)
    false_cnt = len(cmp) - true_cnt

    print(f"Error = {DIFF}")
    print(f"True count {true_cnt}, False = {false_cnt}")
