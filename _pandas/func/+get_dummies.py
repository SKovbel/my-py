import pandas as pd

data = {'Category': ['A', 'B', 'A', 'C', 'B']}
df = pd.DataFrame(data)
df_encoded = pd.get_dummies(df, columns=['Category'], prefix='Category')

print(df)
print(df_encoded)
