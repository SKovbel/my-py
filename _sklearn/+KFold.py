'''
KFold is a technique in cross-validation where the dataset is divided into k subsets, 
    and the model is trained and evaluated k times, 
        each time using a different subset as the test set and the remaining data as the training set
'''

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression  # You can replace this with any other model you want to use
from sklearn import datasets

# Load a sample dataset (e.g., Iris dataset)
iris = datasets.load_iris()
X = iris.data
y = iris.target

model = LogisticRegression()

# Specify the number of folds
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Perform cross-validation
cross_val_results = cross_val_score(model, X, y, cv=kf)

# Display the cross-validation results
print(f"Cross-Validation Scores: {cross_val_results}")
print(f"Mean Cross-Validation Score: {cross_val_results.mean()}")
