import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'family': ['A', 'B', 'C'],
    'value_1': [1, 2, 3],
    'value_2': [4, 5, 6]
}

df = pd.DataFrame(data)

# Applying the operations
le = LabelEncoder()
X = (df
    .set_index('family')  # Setting 'family' as the index
    .stack()  # Wide to long format
    .reset_index('family')  # Convert index to column
    .assign(family=lambda x: le.fit_transform(x.family))  # Label encode 'family'
)
print(df)
print(X)