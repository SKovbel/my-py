import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal
from sklearn.preprocessing import OneHotEncoder

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

class MAFLayer(nn.Module):
    def __init__(self, in_features):
        super(MAFLayer, self).__init__()
        self.in_features = in_features
        self.hidden = 128
        
        self.net = nn.Sequential(
            nn.Linear(in_features, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, in_features * 2)  # output for mean and log variance
        )
    
    def forward(self, x):
        output = self.net(x)
        mean, log_var = output.chunk(2, dim=1)
        return mean, log_var

class MAF(nn.Module):
    def __init__(self, in_features, num_layers=4):
        super(MAF, self).__init__()
        self.layers = nn.ModuleList([MAFLayer(in_features) for _ in range(num_layers)])
    
    def forward(self, x):
        log_det_jacobian = 0
        for layer in self.layers:
            mean, log_var = layer(x)
            z = mean + torch.exp(0.5 * log_var) * torch.randn_like(mean)  # Reparameterization trick
            log_det_jacobian += -0.5 * log_var.sum(dim=1)  # Accumulate log determinant Jacobian
            x = z
        return z, log_det_jacobian

# Create MAF model
in_features = X_train.shape[1]
maf_model = MAF(in_features=in_features)

# Set up the optimizer
optimizer = optim.Adam(maf_model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    z, log_det_jacobian = maf_model(X_train)
    base_distribution = MultivariateNormal(torch.zeros(in_features), torch.eye(in_features))
    log_prob_z = base_distribution.log_prob(z)
    loss = - (log_prob_z + log_det_jacobian).mean()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
maf_model.eval()
with torch.no_grad():
    z_test, log_det_jacobian_test = maf_model(X_test)
    log_prob_z_test = base_distribution.log_prob(z_test)
    likelihood = torch.exp(log_prob_z_test + log_det_jacobian_test)

    print("Test likelihood:", likelihood.mean().item())

def plot_density_estimation(X, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X1, X2 = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X1.ravel(), X2.ravel()]).T
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        z_grid, _ = maf_model(grid_tensor)
        density = torch.exp(base_distribution.log_prob(z_grid)).numpy()
    plt.figure(figsize=(8, 6))
    plt.contourf(X1, X2, density.reshape(X1.shape), levels=30, cmap='Blues', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c='red', s=10)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Density')
    plt.show()

plot_density_estimation(X_test[:, :2].numpy(), "Density Estimation of the Iris Dataset with MAF")
