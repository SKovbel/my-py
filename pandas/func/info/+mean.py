import pandas as pd

data = {'Column1': [10, 15, 20, 25, 30],
        'Column2': [5, 10, 15, 20, 25]}

df = pd.DataFrame(data)

column1_mean = df['Column1'].mean()
column2_mean = df['Column2'].mean()

print(df)

print(f"Mean of Column1: {column1_mean}")
print(f"Mean of Column2: {column2_mean}")
