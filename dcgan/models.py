import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, zdim=10, image_channel=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.zdim = zdim

        self.generator = nn.Sequential(
            self._generator_block(zdim, hidden_dim * 4),
            self._generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self._generator_block(hidden_dim * 2, hidden_dim),
            self._generator_block(hidden_dim, image_channel, kernel_size=4, final_layer=True),
        )

    def _generator_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )
    
    def forward(self, x):
        x = x.view(x.size(0), self.zdim, 1, 1)

        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, image_channels=1, hidden_dim=16):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            self._discriminator_block(image_channels, hidden_dim),
            self._discriminator_block(hidden_dim, hidden_dim * 2),
            self._discriminator_block(hidden_dim * 2, 1, final_layer=True), 
        )
    
    def _discriminator_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )
    
    def forward(self, x):
        x = self.discriminator(x)
        return x.view(x.size(0), -1)
    