import pandas as pd

# Creating two Series
series1 = pd.Series([1, 2, 3, 4], name='A')
series2 = pd.Series([2, 2, 1, 4], name='B')

# Performing element-wise greater-than comparison
result = series1.gt(series2)
result2 = series1.gt(2)

# Displaying the original Series and the result
print(series1)
print(series2)
print("Element-wise Greater Than Comparison Result:")
print(result)
print("Element-wise Greater Than 2:")
print(result2)