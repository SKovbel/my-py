import pandas as pd

# Creating two Series
series1 = pd.Series([1, 2, 3, 4], name='A')
series2 = pd.Series([2, 2, 1, 4], name='B')

# Performing element-wise multiplication
result = series1.mul(series2)

# Displaying the original Series and the result
print("Series 1:")
print(series1)

print("Series 2:")
print(series2)

print("Element-wise Multiplication Result:")
print(result)
