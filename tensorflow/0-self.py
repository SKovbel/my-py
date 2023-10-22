import numpy as np
import tensorflow as tf

x = np.array(((0,0), (0,1), (1,0), (1,1)))
y = np.array([x[i][0] ^ x[i][1] for i in range(len(x))])

net = tf.keras.Sequential()
net.add(tf.keras.layers.Input(shape=(2,)))
net.add(tf.keras.layers.Dense(shape=(2,1), activation="sigmoid"))
net.add(tf.keras.layers.Dense(shape=(2,1), activation="sigmoid"))

loss = tf.keras.losses.binary_crossentropy(y_pred=x, y_true=y)
