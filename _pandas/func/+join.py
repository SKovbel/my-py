import pandas as pd

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'X': ['X0', 'X1', 'X2']},
                   index=['K0', 'K1', 'K2'])

df2 = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                    'D': ['D0', 'D1', 'D2'],
                    'X': ['X2', 'X1', 'X0']},
                   index=['K0', 'K1', 'K2'])

result = df1.join(df2[['C','D']])
result2 = df1.merge(df2[['C', 'D', 'X']], on='X', how='left', suffixes=('_df1', '_df2'))

print(df1)
print(df2)
print(result)
print(result2)
