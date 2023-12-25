import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 15, 20, 25, 30]}

df = pd.DataFrame(data)
grouped_df = df.groupby('Category').mean()

print(df)
print(grouped_df)
