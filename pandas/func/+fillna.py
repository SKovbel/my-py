import pandas as pd
import numpy as np

# Creating a DataFrame with missing values
data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [9, 10, 11, np.nan]}

df = pd.DataFrame(data)
df_filled = df.fillna(0)
df_ffill = df.fillna(method='ffill')

print(df)
print(df_filled)
print(df_ffill)



np.random.seed(0) 

# Create a DataFrame with missing values
data = {'Name': ['Alice', 'Bob', 'Charlie', None, 'Eva'],
        'Age': [25, 30, None, 22, 35],
        'Salary': [50000, 60000, 75000, None, 80000]}

df = pd.DataFrame(data)
df2 = df.fillna('X')

# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
df3 = df.fillna(method='bfill', axis=0).fillna(0)

print('df\n', df)
print('df2\n', df2)
print('df3\n', df3)

