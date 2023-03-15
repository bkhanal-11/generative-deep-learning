import torch
import torch.nn as nn

# Build the autoencoder
class Encoder(nn.Module):
    def __init__(self, channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        
        self.z_mean = nn.Linear(128 * 4 * 4, embedding_dim)
        self.z_log_var = nn.Linear(128 * 4 * 4, embedding_dim)
    
    def _sampling(self, z_mean, z_log_var):
        epsilon = torch.rand_like(z_log_var)
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self._sampling(z_mean, z_log_var)

        return z, z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self, channels, embedding_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.output = nn.Conv2d(32, channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = nn.ReLU()(x)
        x = self.deconv2(x)
        x = nn.ReLU()(x)
        x = self.deconv3(x)
        x = nn.ReLU()(x)
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
