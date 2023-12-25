import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import activations

v = np.arange(-20, 20, 1).astype(float)
x = np.array([v])

print(x) # [[1 2 3 4 5]]

# todo mish, linear, deserialize, exponential, get, hard_sigmoid, serialize, 
print('sigmoid:', keras.layers.Activation('sigmoid')(x))
print('softmax:', keras.layers.Activation('softmax')(x))
print('tanh:', keras.layers.Activation('tanh')(x))
print('softplus:', keras.layers.Activation('softplus')(x))
print('swish:', keras.layers.Activation('swish')(x))
print('softsign:', keras.layers.Activation('softsign')(x))
print('hard_sigmoid:', keras.layers.Activation('hard_sigmoid')(x))

print('linear:', keras.layers.Activation('linear')(x))
print('relu:', keras.layers.Activation(activations.relu)(x))
print('elu:', keras.layers.Activation('elu')(x))
print('selu:', keras.layers.Activation('selu')(x))
print('gelu:', keras.layers.Activation('gelu')(x))

print('mish:', keras.layers.Activation('mish')(x))
print('exponential:', keras.layers.Activation('exponential')(x))
print('lambda:', keras.layers.Activation(lambda x: 2*x)(x))

print('serialize:', keras.layers.Activation('serialize')(x))
#print('deserialize:', keras.layers.Activation('deserialize')(y))

def process(ax, f, x):
    y = keras.layers.Activation(f)(x)
    ax.plot(x.reshape(-1), y.numpy().reshape(-1), label=f)
    ax.legend()
    print(f, '-', x.reshape(-1), y.numpy().reshape(-1))
    
# todo end charts
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, f in enumerate(['sigmoid', 'tanh', 'softmax', 'softsign', 'hard_sigmoid']):
    process(axs[0, 0], f, x)

for i, f in enumerate(['sigmoid', 'softplus', 'swish', 'mish']):
    process(axs[0, 1], f, x)

for i, f in enumerate(['linear', 'relu', 'elu', 'selu', 'gelu']):
    process(axs[1, 0], f, x)

for i, f in enumerate(['exponential']):
    process(axs[1, 1], f, x)

plt.legend()
plt.tight_layout()
plt.show()
