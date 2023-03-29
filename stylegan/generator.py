import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import InstanceNorm2d

class NoiseMappingNetwork(nn.Module):
    def __init__(self, z_dim, hidden_dim, w_dim):
        super(NoiseMappingNetwork, self).__init__()
        
        # Define the hidden layers using nn.ModuleList
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=True),
                nn.ReLU()
            ) for _ in range(6)
        ])
        
        # Define the mapping layers using nn.Sequential
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, hidden_dim, bias=True),
            nn.ReLU()
        )
        
        for i in range(6):
            self.mapping.add_module(f"hidden_{i}", self.hidden_layers[i])
        
        self.mapping.add_module("final_layer", nn.Linear(hidden_dim, w_dim, bias=True))

    def forward(self, noise):
        '''
        Function for completing a forward pass of MappingLayers: 
        Given an initial noise tensor, returns the intermediate noise tensor.
        '''
        return self.mapping(noise)

    
class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter( 
            torch.randn(channels)[None, :, None, None]
        )

    def forward(self, image):
        '''
        Function for completing a forward pass of InjectNoise: Given an image, 
        returns the image with random noise added.
        '''
        # Set the appropriate shape for the noise!
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        
        noise = torch.randn(noise_shape, device=image.device) # Creates the random noise
        return image + self.weight * noise # Applies to image after multiplying by the weight for each channel

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()

        # Normalize the input per-dimension
        self.instance_norm = InstanceNorm2d(channels)

        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        '''
        Function for completing a forward pass of AdaIN: Given an image and intermediate noise vector w, 
        returns the normalized image that has been scaled and shifted by the style.
        '''
        normalized_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        
        transformed_image = style_scale * normalized_image + style_shift

        return transformed_image

class StyleGANGeneratorBlock(nn.Module):

    def __init__(self, in_chan, out_chan, w_dim, kernel_size=3, use_upsample=True):
        super(StyleGANGeneratorBlock, self).__init__()
        self.use_upsample = use_upsample
        
        if self.use_upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size, stride=1, padding=1) # Padding is used to maintain the image size
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size, stride=1, padding=1) # Padding is used to maintain the image size
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)


    def forward(self, x, w):
        '''
        Function for completing a forward pass of MicroStyleGANGeneratorBlock: Given an x and w, 
        computes a StyleGAN generator block.
        '''
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)    
            
        x = self.inject_noise(x)
        x = self.adain(x, w)
        return x

class StyleGANGenerator(nn.Module):
    '''
    StyleGAN Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        map_hidden_dim: the mapping inner dimension, a scalar
        w_dim: the dimension of the intermediate noise vector, a scalar
        channels: a list of channel sizes for each block in the generator
    '''
    def __init__(self, z_dim, map_hidden_dim, w_dim, channels=[512, 256, 128, 64, 32, 16]):
        super(StyleGANGenerator, self).__init__()

        self.num_layers = len(channels) - 1
        self.map = NoiseMappingNetwork(z_dim, map_hidden_dim, w_dim)

        self.initial_block = nn.Parameter(torch.randn(1, channels[0], 4, 4))

        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            block = StyleGANGeneratorBlock(in_channels, out_channels, w_dim, use_upsample=True)  # Fix typo
            self.blocks.append(block)

        self.output_conv = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1)
        self.output_tanh = nn.Tanh()

    def upsample_to_match_size(self, smaller_image, bigger_image):
        '''
        Function for upsampling an image to the size of another: Given a two images (smaller and bigger), 
        upsamples the first to have the same dimensions as the second.
        Parameters:
            smaller_image: the smaller image to upsample
            bigger_image: the bigger image whose dimensions will be upsampled to
        '''
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False, alpha=1.0):
        '''
        Function for completing a forward pass of StyleGANGenerator: Given noise, 
        computes a StyleGAN iteration.
        '''
        x = self.initial_block
        w = self.map(noise)
        intermediates = []

        for i in range(self.num_layers):
            x = self.blocks[i](x, w)
            if i == self.num_layers - 2:
                x_small = x
                x_small_image = self.output_tanh(self.output_conv(x_small))
                intermediates.append(x_small_image)
            if i == self.num_layers - 1:
                x_big = x
                x_big_image = self.output_tanh(self.output_conv(x_big))
                intermediates.append(x_big_image)

        x_small_upsample = self.upsample_to_match_size(intermediates[0], intermediates[1])

        interpolation = alpha * x_big_image + (1 - alpha) * x_small_upsample

        if return_intermediate:
            return interpolation, x_small_upsample, intermediates
        else:
            return interpolation
