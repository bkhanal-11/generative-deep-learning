import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_noise(n_samples, z_dim, device='cpu'):

    return torch.randn(n_samples, z_dim).to(device)

def get_gen_loss(generator, discriminator, criterion, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    fake = generator(noise)
    
    pred_fake = discriminator(fake)
    
    labels_fake = torch.ones_like(pred_fake).to(device)
    gen_loss = criterion(pred_fake, labels_fake)
    
    return gen_loss

def get_disc_loss(generator, disriminator, criterion, real, num_images, z_dim, device):
    noise = get_noise(num_images, z_dim, device)
    fake = generator(noise)
    pred_fake = disriminator(fake.detach())
    labels_fake = torch.zeros_like(pred_fake).to(device)
    
    fake_loss = criterion(pred_fake, labels_fake)
    
    pred_real = disriminator(real)
    labels_real = torch.ones_like(pred_real).to(device)
    real_loss = criterion(pred_real, labels_real)
    
    disc_loss = 0.5 * (real_loss + fake_loss)

    return disc_loss

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()