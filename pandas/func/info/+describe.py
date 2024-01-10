import pandas as pd

# Creating a DataFrame
data = {'Column1': [1, 2, 3, 4],
        'Column2': ['A', 'B', 'C', 'D'],
        'Column3': [1.1, 2.2, 3.3, 4.4]}


df = pd.DataFrame(data)


print(df.describe())