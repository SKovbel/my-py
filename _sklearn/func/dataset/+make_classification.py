from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=100,        # Number of samples
    n_features=2,         # Number of features
    n_informative=2,      # Number of informative features
    n_redundant=0,        # Number of redundant features
    n_classes=2,          # Number of classes
    n_clusters_per_class=1, # Number of clusters per class
    weights=[0.5, 0.5],    # Class distribution
    flip_y=0,             # Probability of flipping the class label
    class_sep=1.0,        # Separation between classes
    random_state=42       # Random seed for reproducibility
)

# Plot the dataset
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Classification Dataset')
plt.show()
