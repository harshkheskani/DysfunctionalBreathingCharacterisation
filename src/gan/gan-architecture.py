import torch
import torch.nn as nn

feature_dim = X_osa_scaled.shape[1]  # Number of features in your 30s window
latent_dim = 100           

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, feature_dim),
            nn.Tanh()  # Tanh outputs values in [-1, 1], matching our scaler
        )

    def forward(self, z):
        return self.model(z)
    
# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Sigmoid outputs values in [0, 1]
        )

    def forward(self, x):
        return self.model(x)