import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
iris = datasets.load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)



# RealNVP, shifting and scaling
class AdditiveCouplingLayer(nn.Module):
    SHUFFLE = True

    def __init__(self, in_features):
        self.in_features = in_features
        super(AdditiveCouplingLayer, self).__init__()
        self.scale = nn.Sequential(
            nn.Linear(in_features // 2, 128),
            nn.ReLU(),
            nn.Linear(128, in_features // 2)
        )
        self.translate = nn.Sequential(
            nn.Linear(in_features // 2, 128),
            nn.ReLU(),
            nn.Linear(128, in_features // 2)
        )
    
    def shuffle_features(self, x):
        perm = torch.randperm(self.in_features)
        return x[:, perm], perm

    def reverse_shuffle_features(self, x, perm):
        x_inv_perm = torch.argsort(perm)
        return x[:, x_inv_perm] 

    def forward(self, x):
        if self.SHUFFLE:
            x, perm = self.shuffle_features(x)

        x_a, x_b = x.chunk(2, dim=1)
        scale = torch.tanh(self.scale(x_a)) 
        translate = self.translate(x_a)
        x_b = x_b * torch.exp(scale) + translate
        x = torch.cat([x_a, x_b], dim=1)

        if self.SHUFFLE:
            x = self.reverse_shuffle_features(x, perm)

        return x, scale


class NormalizingFlow(nn.Module):
    def __init__(self, in_features, num_layers=4):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([AdditiveCouplingLayer(in_features) for _ in range(num_layers)])
    
    def forward(self, x):
        log_det_jacobian = 0
        layer_outputs = [x]
        for layer in self.layers:
            x, scale = layer(x)
            layer_outputs.append(x)
            log_det_jacobian += scale.sum(dim=1)
        return x, log_det_jacobian, layer_outputs


flow_model = NormalizingFlow(in_features=4, num_layers=4)

optimizer = torch.optim.Adam(flow_model.parameters(), lr=0.001)
base_normal_distribution = MultivariateNormal(torch.zeros(4), torch.eye(4))

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    z, log_det_jacobian, _ = flow_model(X_train)
    log_prob_z = base_normal_distribution.log_prob(z)
    loss = -(log_prob_z + log_det_jacobian).mean()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def plot_density_estimation(layer_outputs):
    num_layers = len(layer_outputs)
    fig, axes = plt.subplots(1, num_layers, figsize=(20, 4))
    for i, output in enumerate(layer_outputs):
        output_np = output.detach().numpy()
        x_layer = np.linspace(output_np[:, 0].min() - 1, output_np[:, 0].max() + 1, 100)
        y_layer = np.linspace(output_np[:, 1].min() - 1, output_np[:, 1].max() + 1, 100)
        X_layer, Y_layer = np.meshgrid(x_layer, y_layer)
        grid_points_layer = np.vstack([X_layer.ravel(), Y_layer.ravel()]).T
        kde_layer = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(output_np[:, :2])
        density_layer = np.exp(kde_layer.score_samples(grid_points_layer)).reshape(X_layer.shape)
        axes[i].imshow(density_layer, extent=(x_layer.min(), x_layer.max(), y_layer.min(), y_layer.max()), origin='lower', cmap='viridis', alpha=0.7)
        axes[i].scatter(output_np[:, 0], output_np[:, 1], s=10, color='black', alpha=0.5)
        axes[i].set_title(f'Layer {i}' if i > 0 else 'Input')
        axes[i].set_xlabel('Feature 1')
        axes[i].set_ylabel('Feature 2')

    plt.tight_layout()
    plt.show()

flow_model.eval()
torch.set_printoptions(sci_mode=False, precision=4)
with torch.no_grad():
    z_test, log_det_jacobian_test, layer_outputs = flow_model(X_test)
    log_prob_z_test = base_normal_distribution.log_prob(z_test)
    likelihood = torch.exp(log_prob_z_test + log_det_jacobian_test)
    plot_density_estimation(layer_outputs)
    print("Test likelihood:", likelihood.mean().item())
    combined_print = torch.cat((X_test, z_test), dim=1)
    print(combined_print)