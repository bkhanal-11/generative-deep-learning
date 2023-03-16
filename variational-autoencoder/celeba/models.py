import torch
import torch.nn as nn
import torch.nn.functional as F

# Build the autoencoder
class Encoder(nn.Module):
    def __init__(self, channels, embedding_dim=200, num_features=64, num_layers=4):
        super(Encoder, self).__init__()
        self.conv1 = self.conv_block(channels, num_features)

        self.conv_layers = nn.ModuleList([self.conv_block(num_features, num_features) for _ in range(num_layers)])

        self.flatten = nn.Flatten()
        
        self.z_mean = nn.Linear(256, embedding_dim)
        self.z_log_var = nn.Linear(256, embedding_dim)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def _sampling(self, z_mean, z_log_var):
        epsilon = torch.rand_like(z_log_var)
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
    def forward(self, x):
        x = self.conv1(x)
        
        for layer in self.conv_layers:
            x = layer(x)

        x = self.flatten(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self._sampling(z_mean, z_log_var)

        return z, z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, channels, embedding_dim, num_features=64, num_layers=5):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, 256)
        self.unflatten = nn.Unflatten(1, (64, 2, 2))
        self.deconv_layers = nn.ModuleList([self.deconv_block(num_features, num_features) for _ in range(num_layers - 1)])

        self.output = nn.Sequential(
            self.deconv_block(num_features, num_features),
            nn.Conv2d(num_features, out_channels=channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        
        for layer in self.deconv_layers:
            x = layer(x)

        x = self.output(x)
        return x

class VAE(nn.Module):
    def __init__(self, channels, embedding_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(channels, embedding_dim)
        self.decoder = Decoder(channels, embedding_dim)

    def forward(self, x):
        """Call the model on a particular input."""
        z, z_mean, z_log_var = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z_mean, z_log_var
