from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create a RandomForestClassifier
clf = RandomForestClassifier()

# Use cross_val_score to perform 5-fold cross-validation and get accuracy scores
# Here, 'accuracy' is the scoring metric, and you can change it based on your task
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

# Print the cross-validated accuracy scores
print("Cross-validated Accuracy Scores:", scores)

# Print the mean and standard deviation of the scores
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")
