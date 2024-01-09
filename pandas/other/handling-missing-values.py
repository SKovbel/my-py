import pandas as pd
import numpy as np

np.random.seed(0) 

# Create a DataFrame with missing values
data = {'Name': ['Alice', 'Bob', 'Charlie', None, 'Eva'],
        'Age': [25, 30, None, 22, 35],
        'Salary': [50000, 60000, 75000, None, 80000]}

df = pd.DataFrame(data)
total_missing_counts = df.isnull().sum()

total_cells = np.product(df.shape)
total_missing = df.isnull().sum().sum()
total_missing = total_missing_counts.sum() # or

percent_missing = (total_missing/total_cells) * 100

print('df', df)
print('total_missing_counts', total_missing_counts)
print('total_cells', total_cells)
print('total_missing', total_missing)
print('percent_missing', percent_missing, '%')
