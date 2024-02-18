import pandas as pd

# Create a sample DataFrame with a single column
data = {
    'A': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Squeeze the DataFrame to convert it into a Series
squeezed_series = df.squeeze()
print("\nSqueezed Series:")
print(squeezed_series)