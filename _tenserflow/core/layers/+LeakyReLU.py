import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LeakyReLU

x = np.array([1, 2, 3, 4, 5])
print(x)

# layer
y = LeakyReLU(alpha=0.5)(x)
print(y)


# Model
model = Sequential([
    LeakyReLU(alpha=0.5, input_shape=(5,))
])
y = model.predict(x)
print(y)
