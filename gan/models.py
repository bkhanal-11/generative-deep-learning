import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, zdim=10, image_dim=784, hidden_dim=128):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            self._generator_block(zdim, hidden_dim),
            self._generator_block(hidden_dim, hidden_dim * 2),
            self._generator_block(hidden_dim * 2, hidden_dim * 4),
            self._generator_block(hidden_dim * 4, hidden_dim * 8),
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 8, image_dim),
            nn.Sigmoid(),
        )

    def _generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
    )
    
    def forward(self, x):
        return self.output(self.generator(x))

class Discriminator(nn.Module):
    def __init__(self, image_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            self._discriminator_block(image_dim, hidden_dim * 4),
            self._discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self._discriminator_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1),      
        )
    
    def _discriminator_block(self, input_dim, output_dim):
        return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True),
    )
    
    def forward(self, x):
        return self.discriminator(x)
    