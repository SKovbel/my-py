import pandas as pd
import numpy as np

np.random.seed(0) 

# Create a DataFrame with missing values
data = {'Name': ['Alice', 'Bob', 'Charlie', None, 'Eva'],
        'Age': [25, 30, None, 22, 35],
        'Salary': [50000, 60000, 75000, None, 80000]}

df = pd.DataFrame(data)

print('A', df)
print('B', df.isnull())
print('C', df.isnull().sum())
print('D', df.isnull().sum()[0:1])