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


def rbf_kernel(X1, X2, gamma):
    
    if X1.ndim == 1:
        X1 = X1.reshape(1, -1)
    if X2.ndim == 1:
        X2 = X2.reshape(1, -1)
    
    sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * sq_dists)
  
class SVM:
    def __init__(self, kernel=rbf_kernel, C=1.0, gamma=0.1, tol=1e-3, max_passes=5):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        n_samples, n_features = X.shape

        self.alphas = np.zeros(n_samples)
        self.b = 0
        passes = 0
        
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                E_i = self._decision_function(X[i]) - y[i]
                
                if (y[i] * E_i < -self.tol and self.alphas[i] < self.C) or (y[i] * E_i > self.tol and self.alphas[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    E_j = self._decision_function(X[j]) - y[j]
                    
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * self.kernel(X[i], X[j], gamma=self.gamma) - self.kernel(X[i], X[i], gamma=self.gamma) - self.kernel(X[j], X[j], gamma=self.gamma)
                    if eta >= 0:
                        continue
                    
                    self.alphas[j] -= y[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if np.abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    b1 = self.b - E_i - y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(X[i], X[i], gamma=self.gamma) - y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(X[i], X[j], gamma=self.gamma)
                    b2 = self.b - E_j - y[i] * (self.alphas[i] - alpha_i_old) * self.kernel(X[i], X[j], gamma=self.gamma) - y[j] * (self.alphas[j] - alpha_j_old) * self.kernel(X[j], X[j], gamma=self.gamma)
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0

    def _decision_function(self, X):
        K = self.kernel(X, self.X_train, gamma=self.gamma)
        return np.sum(self.alphas * self.y_train * K) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

class MultiSVM:
    def __init__(self, kernel=rbf_kernel, C=1.0, gamma=0.1, tol=1e-3, max_passes=5):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.svm_models = []

    def fit(self, X, y):
        self.unique_classes = np.unique(y)
        print(self.unique_classes)
        for cls in self.unique_classes:
            y_binary = np.where(y == cls, 1, -1)
            svm = SVM(kernel=self.kernel, C=self.C, gamma=self.gamma, tol=self.tol, max_passes=self.max_passes)
            svm.fit(X, y_binary)
            self.svm_models.append(svm)

    def predict(self, X):
        scores = np.array([svm.predict(X) for svm in self.svm_models])
        return np.argmax(scores, axis=0)



def plt(model, X, y, title):
    plt.figure(figsize=(10, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    
    Z = np.array([model.predict(np.array([a, b])) for a, b in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

model = MultiSVM(C=1.0, gamma=0.1)
model.fit(X_train, y_train)
plt(model, X_train, y_train, "Decision Boundary (Training Set)")
predictions = np.array([model.predict(x) for x in X_test])
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
plt(model, X_test, y_test, "Decision Boundary (Test Set)")
