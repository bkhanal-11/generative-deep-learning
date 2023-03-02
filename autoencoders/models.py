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
        self.fc = nn.Linear(128 * 4 * 4, embedding_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels, embedding_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, 128 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (128, 4, 4))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = nn.ReLU()(x)
        x = self.deconv2(x)
        x = nn.ReLU()(x)
        x = self.deconv3(x)
        x = nn.Sigmoid()(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, channels, embedding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(channels, embedding_dim)
        self.decoder = Decoder(channels, embedding_dim)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
