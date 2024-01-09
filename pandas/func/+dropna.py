import pandas as pd
# remove all the rows that contain a missing value

# Creating a DataFrame
data = {'A': [1, 2, 3, None],
        'B': [5, 6, None, 8],
        'C': [9, 11, 12, 12]}

df = pd.DataFrame(data, index=['row1', 'row2', 'row3', 'row4'])
df_dropped_row = df.dropna()
df_dropped_row = df.dropna(axis=0) # or

# remove all columns with at least one missing value
df_dropped_col = df.dropna(axis=1)


print('A', df)
print('B', df_dropped_row)
print('C', df_dropped_col)

