from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()

# Access the feature matrix and target vector
X = iris.data
y = iris.target

# Display a few rows of the dataset using a DataFrame (optional)
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y
print(iris_df.head())

# Display information about the dataset
print(f"Feature names: {iris.feature_names}")
print(f"Target names: {iris.target_names}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
