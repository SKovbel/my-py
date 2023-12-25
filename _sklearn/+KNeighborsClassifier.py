from sklearn.neighbors import KNeighborsClassifier

# Generate a synthetic classification dataset
#X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X = [[0,0], [1,1], [1,0], [0,1]]
y = [0, 0, 1, 1]

# Create a K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn_classifier.fit(X, y)

# Make predictions on the test set
y_pred = knn_classifier.predict(X)

print(X, y_pred)