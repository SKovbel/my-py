df['Lag_1'] = df['NumVehicles'].shift(1)
df.head()
