import pandas as pd

# Create a DataFrame with missing values
data = {'Name': ['Alice', 'Bob', 'Charlie', None, 'Eva'],
        'Age': [25, 30, None, 22, 35],
        'Salary': [50000, 60000, 75000, None, 80000]}

df = pd.DataFrame(data)

print(df)
print(df.isnull())
print(df.isnull().sum())