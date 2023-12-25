 # The sample method in pandas is used to randomly sample a specified number of rows or a fraction of rows from a DataFrame

import pandas as pd

# Creating a DataFrame
data = {'A': [1, 2, 3, 4, 5],
        'B': ['apple', 'banana', 'cherry', 'date', 'elderberry'],
        'C': [10.1, 20.2, 30.3, 40.4, 50.5]}

df = pd.DataFrame(data)

# Displaying the original DataFrame
print("Original DataFrame:")
print(df)
print()

# Randomly sampling 3 rows from the DataFrame
sampled_df = df.sample(n=3, random_state=11) # 3 rows
sampled_df2 = df.sample(frac=0.4, random_state=11) # 40%

print(sampled_df)
print(sampled_df2)
