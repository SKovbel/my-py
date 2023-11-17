import seaborn as sns
import pandas as pd
from pandas.plotting import andrews_curves
import matplotlib.pyplot as plt


'''
Summary Statistics:
Calculate summary statistics such as mean, median, standard deviation, minimum, and maximum for each numeric feature (e.g., Sepal Length, Sepal Width, Petal Length, and Petal Width).

Correlation:
Compute the correlation matrix to measure the relationships between the numeric features. This can help identify which features are strongly correlated with each other.

Data Distribution:
Create histograms or kernel density plots to visualize the distribution of each numeric feature.

Box Plots:
Use box plots to visualize the distribution of features for each species to see how they differ.

Pair Plots:
Create pair plots to visualize the relationships between features and species. This is a useful tool for exploring the data.

Scatter Plots:
Create scatter plots to explore how pairs of features relate to each other. This can help identify patterns and clusters.

Class Distribution:
Calculate the distribution of species in the dataset. This can help you understand the class balance.

Statistical Testing:
Perform statistical tests to check for significant differences between species in terms of different feature values (e.g., t-tests, ANOVA).

PCA (Principal Component Analysis):
Perform PCA to reduce the dimensionality of the data and visualize the data in lower-dimensional space.

Cluster Analysis:
Apply clustering algorithms to identify natural groupings or clusters within the data.

Regression Analysis:
If applicable, you can explore regression models to predict one feature based on others.

Machine Learning:
Build machine learning models to classify the species based on the feature values. Evaluate the models using various metrics.

Data Visualization:
Use various data visualization techniques, such as scatter plots, bar plots, and heatmaps, to represent the data visually and uncover patterns.
'''
header = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Name"]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_url = pd.read_csv(url, names = header, header=0)

#andrews_curves(iris_url, "Name")
#iris_url.set_index('Name', inplace=True)

print(iris_url.describe())
print(iris_url.groupby("Name").agg(["min", "median", "max", "count"]))
print(iris_url.groupby("Name").corr())

iris_url.hist(bins=20, figsize=(12, 8))
plt.suptitle("Distribution")
plt.show()

#iris_url.plot(kind='kde', figsize=(12, 8))
#plt.title("Density")
#plt.show()

# Create box plots for each numeric feature grouped by species
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i, feature in enumerate(["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]):
    row, col = divmod(i, 2)
    iris_url.boxplot(column=feature, by="Name", ax=axs[row, col])
    axs[row, col].set_title(f"Box Plot of {feature}")
plt.suptitle("Box Plots")
plt.tight_layout()
plt.show()


# Create pair plots
sns.pairplot(iris_url, hue="Name", height=2.5)
plt.suptitle("Pair Plots")
plt.show()


# Create scatter plots for pairs of features
feature_pairs = [("SepalLength", "SepalWidth"), ("PetalLength", "PetalWidth"), ("SepalLength", "PetalLength")]
colors = ["red", "green", "blue"]
plt.figure(figsize=(12, 4))
for i, (x_feature, y_feature) in enumerate(feature_pairs):
    plt.subplot(1, 3, i + 1)
    for species, color in zip(iris_url["Name"].unique(), colors):
        subset = iris_url[iris_url["Name"] == species]
        plt.scatter(subset[x_feature], subset[y_feature], label=species, c=color)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend()
    plt.title(f"Scatter Plot of {x_feature} vs. {y_feature}")
plt.suptitle("Scatter Plotst")
plt.tight_layout()
plt.show()

# Scatter Plots:
# Create scatter plots to visualize the relationship between two numeric features. You can use different colors or markers to represent different species.
sns.scatterplot(data=iris_url, x="SepalLength", y="PetalLength", hue="Name")
plt.title("Scatter Plot of Sepal Length vs. Petal Length")
plt.show()

# Bar Plots:
# Use bar plots to show comparisons between different categories or species for a specific feature.
sns.barplot(data=iris_url, x="Name", y="SepalLength")
plt.title("Average Sepal Length by Species")
plt.show()

# Heatmaps:
# Create heatmaps to visualize correlations between numeric features.
correlation_matrix = iris_url.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")

# Violin Plots:
# Violin plots combine a box plot and a kernel density plot to show the distribution of a numeric feature by category.
sns.violinplot(data=iris_data, x="Name", y="PetalWidth", inner="quart")
plt.title("Distribution of Petal Width by Species")

# Box Plots (Grouped):
# Grouped box plots can help compare the distribution of a numeric feature across different species.
sns.boxplot(data=iris_data, x="Name", y="SepalLength")
plt.title("Distribution Sepal Length")
