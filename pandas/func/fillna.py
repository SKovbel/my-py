import pandas as pd
import numpy as np

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

