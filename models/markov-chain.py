import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

n_bins = 5
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X)

class MarkovChainClassifier:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.transition_probs = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {c: np.mean(y == c) for c in self.classes}
        self.transition_probs = {c: np.zeros((self.n_bins, self.n_bins)) for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            for sample in X_c:
                for i in range(len(sample) - 1):
                    self.transition_probs[c][int(sample[i]), int(sample[i + 1])] += 1
            row_sums = self.transition_probs[c].sum(axis=1)
            self.transition_probs[c] = self.transition_probs[c] / row_sums[:, np.newaxis]

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        log_likelihoods = []
        for c in self.classes:
            log_likelihood = np.log(self.priors[c])
            for i in range(len(x) - 1):
                log_likelihood += np.log(self.transition_probs[c][int(x[i]), int(x[i + 1])])
            log_likelihoods.append(log_likelihood)
        return self.classes[np.argmax(log_likelihoods)]

X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size=0.3, random_state=42)
markov_chain_model = MarkovChainClassifier(n_bins=n_bins)
markov_chain_model.fit(X_train, y_train)

y_pred = markov_chain_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

def plot(X, y, model):
    x0, x1 = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100), np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100))
    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new_discrete = discretizer.transform(X_new)
    y_pred = model.predict(X_new_discrete).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Markov Chain Classifier Decision Boundary')
    plt.show()

plot(X_test, y_test, markov_chain_model)
