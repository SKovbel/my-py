import pandas as pd

# Creating a DataFrame
data = {'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]}

df = pd.DataFrame(data)
column_sums = df.sum(axis=0)
row_sums = df.sum(axis=1)

# Displaying the sums
print(df)
print("Column Sums:")
print(column_sums)

print("Row Sums:")
print(row_sums)
