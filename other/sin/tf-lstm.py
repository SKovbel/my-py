import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from tensorflow.keras import layers, models, optimizers
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt

LR = 0.01
BATCH = 10
EPOCHS = 100

LSTM_IN = 10
LSTM_OUT = 1
LSTM_NODES = 4

MUL = 20
MIN = -20
MAX = 20
MINT = 10
MAXT = 20

x = np.sort(np.random.uniform(MIN, MAX, MUL * (MAX - MIN)))
y = np.sin(x) * np.cos(x/2)
s = y.reshape((len(y), LSTM_OUT))
g = TimeseriesGenerator(s, s, length = LSTM_IN, batch_size = BATCH)
sp = x.reshape((len(x), LSTM_OUT))
gp = TimeseriesGenerator(sp, sp, length = LSTM_IN, batch_size = BATCH)

xt = np.sort(np.random.uniform(MINT, MAXT, MUL * (MAXT - MINT)))
yt = np.sin(xt) * np.cos(xt/2)
st = yt.reshape((len(yt), LSTM_OUT))
gt = TimeseriesGenerator(st, st, length = LSTM_IN, batch_size = BATCH)

model = models.Sequential(layers=[
    layers.LSTM(LSTM_NODES, input_shape=(LSTM_IN, LSTM_OUT)),
    layers.Dense(1)
])

adam = optimizers.Adam(learning_rate=LR)
model.compile(optimizer=adam, loss='mse')

model.fit(gt, epochs=EPOCHS, verbose=1)

yp = model.predict(gp)

plt.plot(x, y, lw=1, c='b', label='sin')
plt.scatter(xt, yt, marker='x', c='g', label='Train')
plt.scatter(x[LSTM_IN:], yp, s=6, c='r', label='Predict')
plt.legend(loc="lower left")

plt.show()