import pandas as pd
# remove all the rows that contain a missing value

# Creating a DataFrame
data = {'A': ['aaa', 'Aaa', 'AAA'],
        'B': ['b', 'b1  ', 'b2  ']}

df = pd.DataFrame(data)
print('A', df)


df['A'] = df['A'].str.lower()
df['B'] = df['B'].str.strip()
print('B', df)

