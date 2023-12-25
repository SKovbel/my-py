import pandas as pd

# Creating a DataFrame
data = {'A': [1, 2, 3, 4],
        'B': [5, 6, 7, 8],
        'C': [9, 10, 11, 12]}

df = pd.DataFrame(data, index=['row1', 'row2', 'row3', 'row4'])
df_dropped_row = df.drop('row2') #, axis=0)

print(df)
print(df_dropped_row)

df_dropped_column = df.drop('B', axis=1)
print(df_dropped_column)
