import numpy as np
import matplotlib.pyplot as plt

train_X = np.linspace(1, 100, 250)
train_y = np.sin(train_X)

def linear_regression(train_X, train_y, pred_X):
    m = len(train_X)
    b = np.c_[np.ones((m, 1)), train_X]
    theta = np.linalg.inv(b.T.dot(b)).dot(b.T).dot(train_y)
    return np.array([1, pred_X]).dot(theta)

def logistic_regression(train_X, train_y, pred_X, learning_rate=0.01, iterations=1000):
    train_X = train_X.reshape(-1, 1)
    m, n = train_X.shape
    b = np.c_[np.ones((m, 1)), train_X]
    theta = np.zeros(n + 1)
    for _ in range(iterations):
        z = np.dot(b, theta)
        h = 1 / (1 + np.exp(-z)) # sigmoid
        cost = -(1/m) * np.sum(train_y * np.log(h) + (1 - train_y) * np.log(1 - h))
        gradient = (1/m) * np.dot(b.T, (h - train_y))
        theta -= learning_rate * gradient
    b = np.array([1, pred_X])
    return 1 / (1 + np.exp(-np.dot(b, theta)))

def locally_weighted_regression(train_X, train_y, pred_X, tau = 1):
    m = train_X.shape[0]
    b = np.c_[np.ones((m, 1)), train_X]
    W = np.zeros(m)
    for i in range(m):
        W[i] = np.exp(-(train_X[i] - pred_X)**2 / (2 * tau**2))
    W = np.diag(W)
    theta = np.linalg.inv(b.T.dot(W).dot(b)).dot(b.T).dot(W).dot(train_y)
    return np.array([1, pred_X]).dot(theta)

pred_X = np.linspace(1, 100, 50)
pred_y0 = np.array([linear_regression(train_X, train_y, x) for x in pred_X])
pred_y1 = np.array([locally_weighted_regression(train_X, train_y, x, tau=0.5) for x in pred_X])
pred_y2 = np.array([logistic_regression(train_X, train_y, x) for x in pred_X])

plt.figure(figsize=(10, 6))
plt.plot(train_X, train_y, color='blue', label='Data Points')
plt.plot(pred_X, pred_y0, color='red', label='LinR')
plt.plot(pred_X, pred_y1, color='orange', label='LocalWeightR')
plt.plot(pred_X, pred_y2, color='cyan', label='LogR')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()