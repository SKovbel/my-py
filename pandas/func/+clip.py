import pandas as pd

# Create a sample DataFrame
data = {
    'A': [1, 5, 10, 15, 20],
    'B': [-3, 0, 7, 12, 18]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Clip the values in the DataFrame to be within the range [0, 10]
clipped_df = df.clip(lower=0, upper=10)
clipped_df2 = df.clip(0.0)
print("\nClipped DataFrame:")
print(clipped_df)
print(clipped_df2)