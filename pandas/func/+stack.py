import pandas as pd

# Create a sample DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Stack the DataFrame
stacked_df = df.stack()
print("\nStacked DataFrame:")
print(stacked_df)
