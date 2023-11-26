import pandas as pd

data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 15, 20, 25, 30]}

df = pd.DataFrame(data)
df['Mean_Value'] = df.groupby('Category')['Value'].transform('mean')

print(df)
print(df)
