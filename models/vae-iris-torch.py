import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

class Encoder(nn.Module):
    def __init__(self, original_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(original_dim, latent_dim)
        self.z_mean = nn.Linear(latent_dim, latent_dim)
        self.z_log_var = nn.Linear(latent_dim, latent_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        h = self.fc(x)
        return self.z_mean(h), self.z_log_var(h)

class Decoder(nn.Module):
    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, original_dim)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, z):
        return self.fc(z)

class VAE(nn.Module):
    def __init__(self, original_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(original_dim, latent_dim)
        self.decoder = Decoder(latent_dim, original_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encoder.forward(x)
        z_std = torch.exp(0.5 * z_log_var)
        z_eps = torch.randn_like(z_std)
        return self.decoder.forward(z_mean + z_std * z_eps), z_mean, z_log_var

    def loss(self, x, z_samples, z_mean, z_log_var):
        rc_loss = nn.functional.mse_loss(z_samples, x, reduction='sum') # Reconstruction loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_log_var.exp() - z_mean.pow(2)) # KL divergence 
        return rc_loss + kl_loss

    def backprop(self, x_train_tensor, epochs = 10000, lr=0.00001):
        self.train()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            z_samples, z_mean, z_log_var = self.forward(x_train_tensor)
            loss = self.loss(x_train_tensor, z_samples, z_mean, z_log_var)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    def reconstruction(self, x_test_tensor):
        self.eval()
        with torch.no_grad():
            x_test_decoded, z_mean, z_log_var = self.forward(x_test_tensor)
        return x_test_decoded

    def generation(self, num=10):
        self.eval()
        z = torch.randn(num, self.encoder.z_mean.out_features)
        return self.decoder.forward(z)

    def anomalies(self, x_test_tensor, threshold=0.01):
        self.eval()
        with torch.no_grad():
            x_test_decoded, z_mean, z_log_var = self.forward(x_test_tensor)
            recon_errors = torch.mean((x_test_tensor - x_test_decoded) ** 2, dim=1)
            anomalies = recon_errors > threshold
            return anomalies, recon_errors

class Dataset:
    def __init__(self, normalize=1, split=0.1):
        self.normalize = normalize
        df = pd.read_csv("data/iris.csv")
        x = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']].values
        self.x_train, self.x_test = train_test_split(x, test_size=split, random_state=1)
        self.norm_mean = self.x_train.mean()
        self.norm_std = self.x_train.std()

    def load(self):
        return (self.normal(torch.tensor(self.x_train, dtype=torch.float32)),
                self.normal(torch.tensor(self.x_test, dtype=torch.float32)))

    def normal(self, x):
        return x if not self.normalize else (x - self.norm_mean) / self.norm_std

    def denormal(self, y):
        return y if not self.normalize else (y * self.norm_std) + self.norm_mean


ds = Dataset(normalize=1)
(x_train_tensor, x_test_tensor) = ds.load()

vae = VAE(original_dim = 4, latent_dim = 2)
vae.backprop(x_train_tensor, epochs = 3000, lr=0.00001)
x_recon = vae.reconstruction(x_test_tensor)
x_anom, x_anom_error = vae.anomalies(x_test_tensor)
x_gener = vae.generation(len(x_test_tensor))

print('Original:', ds.denormal(x_test_tensor))
print('Reconstructed:', ds.denormal(x_recon))
print('Anomalies:', ds.denormal(x_gener), x_anom_error)
print('Generation:', ds.denormal(x_gener))
