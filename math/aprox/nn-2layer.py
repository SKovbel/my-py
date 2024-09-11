import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


x0 = np.linspace(-20, 20, 200).reshape(-1, 1)
y0 = np.sin(x0) 
min_loss = 0.4
max_epochs = 100

tests = [
    #{'l': 1, 'n': 16, 't':1, 's':'-'},
    {'l': 1, 'n': 64, 't':1, 's':'-'},
    {'l': 1, 'n': 128, 't':1, 's':'-'},
    #{'l': 2, 'n': 16, 't':1, 's':'--'},
    {'l': 2, 'n': 64, 't':1, 's':'--'},
    {'l': 2, 'n': 128, 't':1, 's':'--'},
    #{'l': 3, 'n': 16, 't':1, 's':'-.'},
    {'l': 3, 'n': 64, 't':1, 's':'-.'},
    {'l': 3, 'n': 128, 't':1, 's':'-.'},
    #{'l': 4, 'n': 16, 't':1, 's':':'},
    {'l': 4, 'n': 64, 't':1, 's':':'},
    {'l': 4, 'n': 128, 't':1, 's':':'},
]

def create_timeseries(x0, y0, t):
    x, y = [], []
    for i in range(len(x0) - t):
        x.append(x0[i:i + t])
        y.append(y0[i + t])
    return np.array(x), np.array(y)

def create_model(l, n, t):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(t, 1)))
    model.add(tf.keras.layers.Flatten())
    for _ in range(0, l):
        model.add(tf.keras.layers.Dense(n, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x, y, min_loss, max_epochs):
    loss = 0
    for epoch in range(max_epochs):
        history = model.fit(x, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if loss <= min_loss:
            break
    return epoch + 1, loss

plt.figure(figsize=(12, 8))
#plt.plot(x0, y0, label='True', color='blue')

for test in tests:
    l, n, t, s = test['l'], test['n'], test['t'], test['s']
    x, y = create_timeseries(x0, y0, t)
    model = create_model(l, n, t)
    epochs, loss = train_model(model, x, y, min_loss=min_loss, max_epochs=max_epochs)
    print(f'l={l}x{n}, t={t}, epochs={epochs}, loss={loss}')
    y_pred = model.predict(x)
    plt.plot(x0[t:], y_pred, linestyle=s, label=f'l={l}x{n}, t={t}')

plt.legend()
plt.show()


#weights, biases = model.layers[1].get_weights()

#for b in biases:
#    x1, y1 = -10, b
#    x2, y2 = 10, b
#    for w in weights:
#        y1 = y1 + w * x1
#        y2 = y2 + w * x2
#    plt.plot([x1, x2], [y1, y2], color='gray')
