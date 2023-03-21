import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from models import Generator, Discriminator
from utils import *

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set your parameters
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
learning_rate = 0.00001

# Load MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Load Model
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate)

disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate)

criterion = nn.BCEWithLogitsLoss().to(device)

current_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

visualization = False

for epoch in range(n_epochs):
  
    for real, _ in tqdm(train_loader):
        cur_batch_size = real.size(0)

        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to(device)

        ### Update discriminator ###
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim, device)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        ### Update generator ###
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        # Visualization code    
        if current_step % display_step == 0 and current_step > 0:
            writer.add_scalar("Generator Loss | Train", mean_generator_loss, current_step)
            writer.add_scalar("Discriminator Loss | Train", mean_discriminator_loss, current_step)
            print(f"Step {current_step}: \nGenerator loss: {mean_generator_loss} | Discriminator loss: {mean_discriminator_loss}\n")

            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            if visualization:
                show_tensor_images(fake)
                show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        current_step += 1

writer.flush()

# Save the trained model
torch.save(gen, "generator.pth")
