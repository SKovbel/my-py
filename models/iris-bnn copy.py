import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats as stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class BayesianLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.ones(out_features, in_features) * -5)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_sigma = nn.Parameter(torch.ones(out_features) * -5)

    def forward(self, x):
        bias_sigma = torch.exp(self.bias_sigma)
        weight_sigma = torch.exp(self.weight_sigma)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        return F.linear(x, weight, bias)

class BayesianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = BayesianLayer(4, 16)
        self.layer2 = BayesianLayer(16, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

    def kl_divergence(self, mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu.pow(2) - torch.exp(sigma))

    def train_model(self, X_train, y_train, num_epochs=1000, kl_weight=0.1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bnn.parameters(), lr=0.01)

        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(X_train)
            nl_loss = criterion(outputs, y_train)
            kl_loss = (self.kl_divergence(self.layer1.weight_mu, self.layer1.weight_sigma) +
                       self.kl_divergence(self.layer2.weight_mu, self.layer2.weight_sigma))
            total_loss = nl_loss + kl_weight * kl_loss
            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss.item()}')

    def predict(self, X_test, num_samples=1):
        self.eval()
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self(X_test)
                preds.append(outputs.cpu().numpy())
        return np.array(preds)
        #mean_preds = np.mean(all_preds, axis=0)
        #preds = torch.argmax(outputs, dim=1).cpu().numpy()



iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bnn = BayesianNN()
bnn.train_model(X_train, y_train, num_epochs=1000)
predictions = bnn.predict(X_test)
means = np.mean(predictions, axis=0)
#print(predictions)
print(means)
#for pred, mean, y_true in zip(predictions, means, y_test):
#    print(f"Predicted Samples: {pred}, Mean Prediction: {mean:.2f}, True: {y_true}")
