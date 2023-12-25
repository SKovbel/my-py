import pandas as pd

df = pd.DataFrame({"A": ["Word1, Word2. Word3 Word4"]})


df['A1'] = df['A'].apply(lambda x: x.split()[1])
df['A2'] = df['A'].apply(lambda x: x.split(',')[0]).apply(lambda x: x.split()[0])


print(df)