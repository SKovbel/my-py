'''
 The silhouette score is a metric used to calculate the goodness of a clustering technique.
 It measures how well-defined the clusters are in a given clustering arrangement.
 The silhouette score ranges from -1 to 1,
   where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters
'''

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data

# Apply KMeans clustering with different numbers of clusters
for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Calculate the silhouette score
    silhouette_avg = silhouette_score(X, labels)
    
    print(f"For n_clusters={n_clusters}, the silhouette score is {silhouette_avg:.2f}")
