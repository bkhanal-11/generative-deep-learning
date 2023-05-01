import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import VQVAE

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 256
EPOCHS = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

learning_rate = 1e-3

# Dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
    ])

training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

data_variance = np.var(training_data.data / 255.0)

training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)

validation_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)


# Define a model
model = VQVAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_recon_error = []
train_total_loss = []

for i in range(EPOCHS):
    (data, _) = next(iter(training_loader))
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()
    
    train_recon_error.append(recon_error.item())
    train_total_loss.append(loss.item())

    if (i + 1) % 100 == 0:
        print(f'Epoch {i+1}: Reconstruction Error: {np.mean(train_recon_error[-100:])} | Traing Loss: {np.mean(train_total_loss[-100:])}')

torch.save(model, 'vq_vae.pt')

