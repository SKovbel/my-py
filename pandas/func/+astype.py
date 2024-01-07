import pandas as pd

data = {'Column1': [1, 2, 3, 4],
        'Column2': ['A', 'B', 'C', 'D'],
        'Column3': [1.1, 2.2, 3.3, 4.4]}

df = pd.DataFrame(data)

print(df)
print(df.dtypes)

df['Column1'] = df['Column1'].astype(float)
df['Column3'] = df['Column3'].astype(int)


print(df)
print(df.dtypes)
