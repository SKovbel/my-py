from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

# Sample categorical data
categories = np.array(['cat', 'dog', 'bird', 'cat', 'dog']).reshape(-1, 1)

# Create OneHotEncoder instance
encoder = OneHotEncoder()

# Fit and transform the data
one_hot_encoded = encoder.fit_transform(categories).toarray()

# Get the feature names for the one-hot encoded columns
feature_names = encoder.get_feature_names_out(input_features=['category'])

# Create a DataFrame with the one-hot encoded data and feature names
df_one_hot = pd.DataFrame(one_hot_encoded, columns=feature_names)

# Print the DataFrame
print(categories)
print(feature_names)
print(df_one_hot)



'''
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
...
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
...
X_train = X_train.drop(object_cols, axis=1)
X_train = pd.concat([X_train, OH_cols_valid], axis=1)
'''