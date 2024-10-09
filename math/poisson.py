import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

x = np.linspace(-20, 20, 100)

y_linear = x
y_logistic = expit(x)
y_poisson = np.exp(x)

plt.figure(figsize=(10, 6))

plt.plot(x, y_linear, label='Linear Regression', color='blue', linestyle='-')
plt.plot(x, y_logistic, label='Logistic Regression', color='green', linestyle='--')
plt.plot(x, y_poisson, label='Poisson Regression', color='red', linestyle='-.')
plt.ylim(-5, 15)

plt.xlabel('x')
plt.ylabel('Predicted Value')
plt.title('Linear vs Logistic vs Poisson Regression')
plt.legend()
plt.grid(True)
plt.show()