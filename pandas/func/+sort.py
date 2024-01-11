import pandas as pd
# remove all the rows that contain a missing value

# Creating a DataFrame
data = {'A': [1, 6, 3, 2],
        'B': [5, 6, None, 8],
        'C': [9, 11, 12, 12]}

df = pd.DataFrame(data, index=['row1', 'row2', 'row3', 'row4'])
a = df['A'].unique()
a.sort()
print('B', a)

