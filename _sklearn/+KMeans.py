from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, max_iter=100, n_init=5, random_state=10)
cluster = kmeans.fit(X_scaled)
print(cluster)
labels = kmeans.labels_

iris_with_clusters = iris.copy()
iris_with_clusters['Cluster'] = labels

# Evaluating the clustering performance using silhouette score
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette_avg}")

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue='Cluster', data=iris_with_clusters, palette="Set1", s=100)
plt.title("KMeans Clustering of Iris Dataset")
plt.xlabel("Feature 0 (Standardized)")
plt.ylabel("Feature 1 (Standardized)")
plt.show()
