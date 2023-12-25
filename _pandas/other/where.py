import pandas as pd

df = pd.DataFrame({'A': ['A1', 'A1', 'A2'],
                    'B': ['B0', 'B1', 'B2'],
                    'X': [3, 2, 1]},
                   index=['K0', 'K1', 'K2'])

result = df[((df['A'] == 'A1') & (df['X'] > 2)) | (df['B'] == 'B2')]

print(result)
