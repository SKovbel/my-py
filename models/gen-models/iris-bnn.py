import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

'''
mu = [0,0,0,0,0,0,...]
sigma = [-5, -5, -5, -5, -5, -5,...]

forward:
    w = mu + e^sigma * rand()
    b = mu + e^sigma * rand()
    y = xw + b
'''

class BayesianLayer(nn.Module):
    def __init__(self, in_num, out_num):
        super().__init__()
        self.w_mu = nn.Parameter(torch.zeros(out_num, in_num))
        self.b_mu = nn.Parameter(torch.zeros(out_num))
        self.w_sigma = nn.Parameter(torch.ones(out_num, in_num) * -5)
        self.b_sigma = nn.Parameter(torch.ones(out_num) * -5)

    def forward(self, x):
        b_sigma = torch.exp(self.b_sigma)
        w_sigma = torch.exp(self.w_sigma)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        return nn.functional.linear(x, w, b)

class BayesianNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = BayesianLayer(4, 16)
        self.layer_2 = BayesianLayer(16, 3)

    def forward(self, x):
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        return x

    def divergence_kl(self, mu, sigma):
        return -0.5 * torch.sum(1 + sigma - mu.pow(2) - torch.exp(sigma))

    def train_model(self, X_train, y_train, num_epochs=1000, kl_weight=0.1):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(bnn.parameters(), lr=0.01)

        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = self(X_train)
            loss_nl = criterion(outputs, y_train)
            loss_kl = (self.divergence_kl(self.layer_1.w_mu, self.layer_1.w_sigma) +
                       self.divergence_kl(self.layer_2.w_mu, self.layer_2.w_sigma))
            loss_total = loss_nl + kl_weight * loss_kl
            loss_total.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss_total.item()}')

    def predict(self, X_test, num_samples=1):
        self.eval()
        preds = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self(X_test)
                preds.append(outputs.cpu().numpy())
        return np.array(preds)

iris = load_iris()
X = torch.tensor(iris.data, dtype=torch.float32)
y = torch.tensor(iris.target, dtype=torch.long)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bnn = BayesianNN()
bnn.train_model(X_train, y_train, num_epochs=1000)
predictions = bnn.predict(X_test)

y_preds = np.argmax(predictions, axis=2).reshape(-1, 1)
for prediction, y_pred, y_true in zip(predictions[0], y_preds.flatten(), y_test):
    print(f"Prediction: {y_pred:.2f}, True: {y_true}, {prediction}")
