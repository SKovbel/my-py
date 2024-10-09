import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400).reshape(-1, 1)
y = x**2
n = 5
epochs = 100

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1)),
    tf.keras.layers.Dense(n, use_bias=True),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, epochs=epochs, verbose=0)
y_pred = model.predict(x)
print(y_pred)

plt.figure(figsize=(12, 8))
weights, biases = model.layers[1].get_weights()

for w in weights:
    for b in biases:
        x1, y1 = -10, w * -10 + b
        x2, y2 = 10, w * 10 + b
        plt.plot([x1, x2], [y1, y2], color='gray', label=f'{w} {b}')

plt.plot(x, y, label='True Parabola: $y = x^2$', color='blue', linestyle='--')
plt.plot(x, y_pred, label='Fitted Parabola by Model', color='red')

plt.scatter(x, y, color='black', s=10, label='True Data Points')

plt.title('Quadratic Function Approximation using TensorFlow Dense Layer')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
