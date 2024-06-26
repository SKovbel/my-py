import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.random.randn(100).cumsum()
df = pd.DataFrame(data, index=dates, columns=['Value'])

print(df.head())

window_size = 7

df['RollingMean'] = df['Value'].rolling(window=window_size).mean()
df['RollingStd'] = df['Value'].rolling(window=window_size).std()
print(df.head(15))


plt.figure(figsize=(12, 6))
plt.plot(df['Value'], label='Original Data')
plt.plot(df['RollingMean'], label=f'{window_size}-Day Rolling Mean', color='orange')
plt.fill_between(df.index, df['RollingMean'] - df['RollingStd'], df['RollingMean'] + df['RollingStd'], color='orange', alpha=0.2)
plt.legend()
plt.title('Trend Rollup using Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()