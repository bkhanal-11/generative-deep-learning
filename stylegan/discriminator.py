import torch
import torch.nn as nn

class StyleGANDiscriminatorBlock(nn.Module):
    '''
    StyleGAN Discriminator Block Class
    Values:
        in_chan: the number of channels in the input feature map
        out_chan: the number of channels in the output feature map
    '''

    def __init__(self, in_chan, out_chan):
        super(StyleGANDiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1, groups=in_chan)
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1)
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        '''
        Function for completing a forward pass of StyleGANDiscriminatorBlock: Given an x, 
        computes a StyleGAN discriminator block.
        '''
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.downsample(x)
        return x

class StyleGANDiscriminator(nn.Module):
    '''
    StyleGAN Discriminator Class
    Values:
        channels: a list of channel sizes for each block in the discriminator
    '''

    def __init__(self, channels=[16, 32, 64, 128, 256, 512]):
        super(StyleGANDiscriminator, self).__init__()

        self.num_layers = len(channels)
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            in_channels = channels[i] if i == 0 else channels[i-1]
            out_channels = channels[i]
            block = StyleGANDiscriminatorBlock(in_channels, out_channels)
            self.blocks.append(block)

        self.final_conv = nn.Conv2d(channels[-1], 1, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        Function for completing a forward pass of StyleGANDiscriminator: Given an x, 
        computes a StyleGAN discriminator output.
        '''
        for i in range(self.num_layers):
            x = self.blocks[i](x)
        x = self.final_conv(x)
        return x.squeeze()
