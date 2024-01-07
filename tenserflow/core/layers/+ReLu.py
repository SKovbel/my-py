import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import ReLU

x = np.array([1, 2, 3, 4, 5])
print(x)

# layer
y = ReLU(threshold=0.5, max_value=4.0, negative_slope=0.0)(x)
print(y)


# Model
model = Sequential([
    ReLU(input_shape=(5,))
])
y = model.predict(x)
print(y)
