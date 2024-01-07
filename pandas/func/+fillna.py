import pandas as pd
import numpy as np

# Creating a DataFrame with missing values
data = {'A': [1, 2, np.nan, 4],
        'B': [5, np.nan, 7, 8],
        'C': [9, 10, 11, np.nan]}

df = pd.DataFrame(data)
df_filled = df.fillna(0)
df_ffill = df.fillna(method='ffill')

print(df)
print(df_filled)
print(df_ffill)
