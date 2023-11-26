import pandas as pd
'''
The transform function in pandas is used to perform some operation on a grouped data and then broadcast
    the result back to the original DataFrame.
'''
data = {'Category': ['A', 'B', 'A', 'B', 'A'],
        'Value': [10, 15, 20, 25, 30]}

df = pd.DataFrame(data)

df['Mean_Value'] = df.groupby('Category')['Value'].transform('mean')
df['Range_Value'] = df.groupby('Category')['Value'].transform(lambda x: x.max() - x.min())

print(df)
