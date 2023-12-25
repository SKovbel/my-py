import tensorflow as tf
from tensorflow import keras
import numpy as np

# mean_i = sum(x_i[j] for j in range(k)) / k
# var_i = sum((x_i[j] - mean_i) ** 2 for j in range(k)) / k
# x_i_normalized = (x_i - mean_i) / sqrt(var_i + epsilon)
# output_i = x_i_normalized * gamma + beta

model = keras.Sequential([
    keras.layers.LayerNormalization()
])

x = np.array([
    [0.0, 5.0, 10.0]
])
y = model.predict(x)

#[[-1.2  0.0  1.2]]
print(y)
