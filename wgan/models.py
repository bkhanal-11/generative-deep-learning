import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=10, image_channels=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.generator = nn.Sequential(
            self._generator_block(z_dim, hidden_dim * 4),
            self._generator_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self._generator_block(hidden_dim * 2, hidden_dim),
            self._generator_block(hidden_dim, image_channels, kernel_size=4, final_layer=True),
        )

    def _generator_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        
        return self.generator(x)

class Critic(nn.Module):
    def __init__(self, image_channels=1, hidden_dim=64):
        super(Critic, self).__init__()
        
        self.critic = nn.Sequential(
            self._critic_block(image_channels, hidden_dim),
            self._critic_block(hidden_dim, hidden_dim * 2),
            self._critic_block(hidden_dim * 2, 1, final_layer=True),
        )

    def _critic_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        crit_pred = self.critic(image)
        
        return crit_pred.view(len(crit_pred), -1)