import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Create a MinMaxScaler
scaler = MinMaxScaler()

test = np.array([1,2,3,4,5,6,7,8,10]).reshape(-1, 1)
test_scaled = scaler.fit_transform(test)
test_scaled2 = scaler.transform(test)

print(test)
print(test_scaled)
print(test_scaled2)