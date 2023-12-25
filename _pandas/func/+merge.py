import pandas as pd

# Creating two DataFrames
df1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                    'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3']})

df2 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                    'B': ['B0', 'B1', 'B2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})

result = pd.merge(df1, df2, on='key')

print(df1)
print(df2)
print(result)
