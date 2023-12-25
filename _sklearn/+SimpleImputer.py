import numpy as np
from sklearn.impute import SimpleImputer

# Create a sample dataset with missing values
X = np.array([[1, 2, np.nan],
              [4, np.nan, 6],
              [7, 8, 9]])

# Instantiate the SimpleImputer with strategy='mean'
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the dataset and transform it
X_imputed = imputer.fit_transform(X)

# Print the imputed dataset
print("Original dataset with missing values:")
print(X)
print("\nImputed dataset:")
print(X_imputed)
