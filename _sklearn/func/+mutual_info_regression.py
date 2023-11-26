import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# Generate some random data
np.random.seed(42)
X = pd.DataFrame({
    'A': [1, 2, 2, 4, 1, 4, 5, 6, 7],
    'B': [4, 3, 2, 1, 1, 2, 2, 1, 0],
    'R': [1, 0, 0, 1, 1, 0, 0, 1, 1]
})
y = X["R"]  # Convert y to a pandas Series

def make_mi_scores(X, y, discrete_features):
    print(X, y)
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

discrete_features = X.dtypes == 'int64'
print(discrete_features)
mi_scores = make_mi_scores(X, y, discrete_features)
print(mi_scores)  # Show a few features with their MI scores

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores)
plt.show()


features = ["A", "B"]
sns.relplot(
    x="value", y="R", col="variable", data=X.melt(id_vars="R", value_vars=features), facet_kws=dict(sharex=False),
);
plt.show()


sns.catplot(x="A", y="R", data=X, kind="boxen")
plt.show()

