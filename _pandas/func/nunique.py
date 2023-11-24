import pandas as pd

# Create a sample DataFrame
data = {'Category': ['A', 'B', 'C', 'A', 'B', 'D']}
df = pd.DataFrame(data)

# Count the number of unique values in the 'Category' column
unique_count = df['Category'].nunique()

print(f'Number of unique values in the "Category" column: {unique_count}')


'''
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
'''