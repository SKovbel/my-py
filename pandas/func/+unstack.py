import pandas as pd

# Create a sample DataFrame
data = {
    ('A', 'X'): [1, 2, 3],
    ('A', 'Y'): [4, 5, 6],
    ('B', 'X'): [7, 8, 9],
    ('B', 'Y'): [10, 11, 12]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Unstack the DataFrame
unstacked_df = df.unstack()
print("\nUnstacked DataFrame:")
print(unstacked_df)