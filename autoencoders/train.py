import torch
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
import torch.nn as nn
import numpy as np

from models import Autoencoder

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 100
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 3

transform = Compose([
    Resize((32, 32)),  # Resize to 32x32
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5])
])

train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

autoencoder = Autoencoder(channels=CHANNELS, embedding_dim=EMBEDDING_DIM)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

autoencoder.to(device)
loss_fn.to(device)

# Train the model
for epoch in range(EPOCHS):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        reconstructed_data = autoencoder(data)

        loss = loss_fn(reconstructed_data, data)

        loss.backward()

        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch: {} Batch: {} Loss: {:.6f}'.format(
                epoch, batch_idx, loss.item()))

# Evaluate the model
autoencoder.eval()

with torch.no_grad():
    test_loss = 0
    for data, _ in test_loader:
        data = data.to(device)

        reconstructed_data = autoencoder(data)

        test_loss += loss_fn(reconstructed_data, data).item()

    test_loss /= len(test_loader.dataset)
    print('Test set loss: {:.6f}'.format(test_loss))

# Save the model checkpoint
torch.save(autoencoder, 'autoencoder.pth')
