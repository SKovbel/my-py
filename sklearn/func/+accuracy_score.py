from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

X = [10,10,20]
y = [10,30,20]

# Calculate the mean squared error
accuracy = accuracy_score(X, y)
print(f'Accuracy score: {accuracy}')
