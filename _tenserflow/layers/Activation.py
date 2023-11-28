import tensorflow as tf
from tensorflow import keras
import numpy as np

x = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0]
])

print(x) # [[1 2 3 4 5]]

# todo mish, linear, deserialize, exponential, get, hard_sigmoid, serialize, 
print('sigmoid:', keras.layers.Activation('sigmoid')(x))
print('softmax:', keras.layers.Activation('softmax')(x))
print('tanh:', keras.layers.Activation('tanh')(x))
print('softplus:', keras.layers.Activation('softplus')(x))
print('swish:', keras.layers.Activation('swish')(x))
print('softsign:', keras.layers.Activation('softsign')(x))
print('linear:', keras.layers.Activation('linear')(x))

print('relu:', keras.layers.Activation('relu')(x))
print('elu:', keras.layers.Activation('elu')(x))
print('selu:', keras.layers.Activation('selu')(x))
print('gelu:', keras.layers.Activation('gelu')(x))

print('mish:', keras.layers.Activation('mish')(x))
print('exponential:', keras.layers.Activation('exponential')(x))
print('get:', keras.layers.Activation(lambda x: 100*x)(x))
print('hard_sigmoid:', keras.layers.Activation('hard_sigmoid')(x))

y = keras.layers.Activation('serialize')(x)
print('serialize:', y)
#print('deserialize:', keras.layers.Activation('deserialize')(y))
