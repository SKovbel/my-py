import pandas as pd

print("\nCONCAT")
print("Concatenation is used to combine two or more data structures along a particular axis.")
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
print("Concatenate along rows (axis=0)")
result = pd.concat([df1, df2])
print('A=\n', df1)
print('B=\n', df2)
print("CONCAT\n", result)

print("\nJOIN")
print("Joining is used to combine columns from two different DataFrames.")
df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'value': [3, 4]})
print("Merge or join on the 'key' column")
result = pd.merge(df1, df2, on='key')
print("JOIN", result)

print("\nMERGE")
print("Merge is similar to join, and it is used to combine two DataFrames based on a common column.")
df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'value': [3, 4]})
print("Merge on the 'key' column")
result = pd.merge(df1, df2, on='key')
print("MERGE ", result)

print("\nMAP")
print("Mapping is used to apply a function to each element of a Series.")
df = pd.DataFrame({'A': [1, 2, 3]})
print("Map a function to the 'A' column")
df['B'] = df['A'].map(lambda x: x * 2)
print("MAP", df)

print("\nINDEXES []")
print("Creating New DataFrame/Series:")
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("Create a new DataFrame")
new_df = df[df['A'] > 1]
print("Create a new Series")
new_series = pd.Series([7, 8, 9], name='C')
print("NEW DATAFRAME ", new_df)
print("NEW_SERIES ", new_series)

print("\nAPPLY")
print("The apply function can be used to apply a function along the axis of a DataFrame or Series.")
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result_apply = df.apply(lambda col: col * 2)
print("APPLY", result_apply)

print("\nORDER")
print("Ordering a DataFrame by a specific column:")
df = pd.DataFrame({'A': [3, 1, 2], 'B': [6, 4, 5]})
print("Order the DataFrame by column 'A'")
result_order = df.sort_values(by='A')
print("ORDER", result_order)

print("\nGROUPBY")
print("Grouping by a specific column and applying an aggregation function:")
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar'],
                   'B': [1, 2, 3, 4]})
print("Group by column 'A' and calculate the mean for each group")
result_groupby = df.groupby('A').mean()
print("GROUPBY", result_groupby)


help(round(-2.01))






