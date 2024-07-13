import pandas as pd
import numpy as np

np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', end='2023-01-14', freq='D')
data = np.random.randint(1, 100, size=len(date_range))
df = pd.DataFrame({'date': date_range, 'sales': data})

df.set_index('date', inplace=True)

df_weekly = df.resample('W-MON  ').sum()

df_weekly.reset_index(inplace=True)

print(df)
print(df_weekly)
