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
n_epochs = 50
zdim = 64
display_step = 500
batch_size = 128
learning_rate = 0.0002

beta_1 = 0.5 
beta_2 = 0.999

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Load MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Load Model
gen = Generator(zdim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate, betas=(beta_1, beta_2))

criterion = nn.BCEWithLogitsLoss().to(device)

# Initialize the weights to the normal distribution
# with mean 0 and standard deviation 0.02
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

# Train models
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

visualization = False

for epoch in range(n_epochs):
    for real, _ in tqdm(train_loader):
        cur_batch_size = real.size(0)
        real = real.to(device)

        ## Update discriminator ##
        disc_opt.zero_grad()
        fake_noise = get_noise(cur_batch_size, zdim, device=device)
        fake = gen(fake_noise)
        disc_fake_pred = disc(fake.detach())
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_pred = disc(real)
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward(retain_graph=True)
        # Update optimizer
        disc_opt.step()

        ## Update generator ##
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, zdim, device=device)
        fake_2 = gen(fake_noise_2)
        disc_fake_pred = disc(fake_2)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        ## Visualization code ##
        if cur_step % display_step == 0 and cur_step > 0:
            writer.add_scalar("Generator Loss | Train", mean_generator_loss, cur_step)
            writer.add_scalar("Discriminator Loss | Train", mean_discriminator_loss, cur_step)
            print(f"Epoch {epoch + 1}, Step {cur_step}: \nGenerator loss: {mean_generator_loss} | Discriminator loss: {mean_discriminator_loss}\n")
            if visualization:
                show_tensor_images(fake)
                show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1

writer.flush()

# Save the trained model
torch.save(gen, "dcgan_generator.pth")
