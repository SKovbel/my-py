import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d  # noqa: F401

def train_and_predict(train_input_features, train_outputs, prediction_features):
    """
    :param train_input_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains: sepal length, sepal width, petal length, and petal width   
    :param train_outputs: (numpy.array) A one-dimensional NumPy array where each element
                        is a number representing the species of iris which is described in
                        the same row of train_input_features. 0 represents Iris setosa,
                        1 represents Iris versicolor, and 2 represents Iris virginica.
    :param prediction_features: (numpy.array) A two-dimensional NumPy array where each element
                        is an array that contains: sepal length, sepal width, petal length, and petal width
    :returns: (list) The function should return an iterable (like list or numpy.ndarray) of the predicted 
                        iris species, one for each item in prediction_features
    """   
    svm_classifier = SVC(kernel='linear', C=1)
    svm_classifier.fit(train_input_features, train_outputs)
    predictions = svm_classifier.predict(prediction_features)
    return predictions


iris = datasets.load_iris()

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.3, random_state=0)

y_pred = train_and_predict(X_train, y_train, X_test)
if y_pred is not None:
    print(metrics.accuracy_score(y_test, y_pred))

def print_3d():
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=3).fit_transform(iris.data)
    ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    ax.set_title("First three PCA dimensions")
    ax.set_xlabel("Setosa")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("Versicolor")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("Virginica")
    ax.zaxis.set_ticklabels([])

    plt.show()


print_3d()