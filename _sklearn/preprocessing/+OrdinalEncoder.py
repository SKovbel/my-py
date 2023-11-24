from sklearn.preprocessing import OrdinalEncoder

# Sample data with a categorical feature
data = [['Low'], ['Medium'], ['High'], ['Low'], ['Medium']]

# Create an OrdinalEncoder instance
encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])

# Fit and transform the data
encoded_data = encoder.fit_transform(data)

print(encoded_data)