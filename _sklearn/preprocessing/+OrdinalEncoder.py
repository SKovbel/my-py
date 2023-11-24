from sklearn.preprocessing import OrdinalEncoder

# Sample data with a categorical feature
data = [['Low'], ['Medium'], ['High'], ['Low'], ['Medium']]
labels = [['cat'], ['dog'], ['bird'], ['cat'], ['dog']]

# Create an OrdinalEncoder instance
encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
encoder2 = OrdinalEncoder(categories=[['cat', 'dog', 'bird']])

# Fit and transform the data
encoded_data = encoder.fit_transform(data)
print(encoded_data)

encoded_data = encoder2.fit_transform(labels)
print(encoded_data)


'''
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
good_label_cols = [col for col in object_cols if set(X_valid[col]).issubset(set(X_train[col]))]
bad_label_cols = list(set(object_cols)-set(good_label_cols))

label_X_valid = X_valid.drop(bad_label_cols, axis=1)
encoder = OrdinalEncoder()
label_X_train[good_label_cols] = encoder.fit_transform(X_train[good_label_cols])
label_X_valid[good_label_cols] = encoder.transform(X_valid[good_label_cols])
'''