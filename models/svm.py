import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class SVM:
    def __init__(self, learning_rate=0.001, alpha=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        _, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.epochs):
            for i in range(len(y)):
                if y[i] * (np.dot(X[i], self.w) + self.b) >= 1:
                    self.w -= self.learning_rate * 2 * self.alpha * self.w
                else:
                    self.w -= self.learning_rate * (2 * self.alpha * self.w - np.dot(X[i], y[i]))
                    self.b -= self.learning_rate * self.alpha * y[i]

    def predict(self, X):
        print(len(X))
        return np.sign(np.dot(X, self.w) - self.b)

class MultiSVM:
    def __init__(self, learning_rate=0.001, alpha=0.01, epochs=1000):
        self.lr = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.svm_models = []

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        print(self.unique_classes)
        for cls in self.unique_classes:
            y_binary = np.where(y == cls, 1, -1)
            svm = SVM(learning_rate=self.lr, alpha=self.alpha, epochs=self.epochs)
            svm.fit(X, y_binary)
            self.svm_models.append(svm)

    def predict(self, X):
        scores = np.array([svm.predict(X) for svm in self.svm_models])
        return np.argmax(scores, axis=0)

def plot(X, y, model):
    x0, x1 = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred = model.predict(X_new).reshape(x0.shape)
    print(y_pred)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Multiclass Decision Boundary')
    plt.show()

svm = MultiSVM(learning_rate=0.001, alpha=0.01, epochs=2)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

plot(X_test, y_test, svm)
