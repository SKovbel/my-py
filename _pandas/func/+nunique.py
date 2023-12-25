import pandas as pd

data = {'Category': ['A', 'B', 'A', 'C', 'B'],
        'Value': [10, 15, 20, 15, 10]}

df = pd.DataFrame(data)
unique_values_category = df['Category'].nunique()
unique_values_value = df['Value'].nunique()

print(df)
print(df.nunique())

print(f"Number of unique values in 'Category': {unique_values_category}")
print(f"Number of unique values in 'Value': {unique_values_value}")
