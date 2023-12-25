import pandas as pd

# Assuming 'data' is your original DataFrame

# Convert 'recorded_time' to datetime and extract features
data['recorded_time'] = pd.to_datetime(data['recorded_time'])
data['dayofyear'] = data['recorded_time'].dt.dayofyear
data['hour'] = data['recorded_time'].dt.hour

# Convert 'humidity' to float and replace NaN with 0.0
data['humidity'] = pd.to_numeric(data['humidity'], errors='coerce').fillna(0.0)

# Assuming you want to drop 'power_output', 'recorded_time' from the features
train_x = data.drop(['power_output', 'recorded_time'], axis=1)
train_y = data['power_output']
