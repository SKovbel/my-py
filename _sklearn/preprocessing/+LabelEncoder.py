from sklearn.preprocessing import LabelEncoder

# Sample data with categorical labels
labels = ['cat', 'dog', 'bird', 'cat', 'dog']

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels to integer values
encoded_labels = label_encoder.fit_transform(labels)

# Display the original labels and their encoded values
for label, encoded_value in zip(labels, encoded_labels):
    print(f"{label}: {encoded_value}")

# Inverse transform to get back the original labels from encoded values
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print("\nDecoded Labels:", decoded_labels)
