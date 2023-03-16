import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import VAE
from torchsummary import summary

# Define hyperparameters
batch_size = 128
learning_rate = 0.0005
num_epochs = 10
r_loss_factor = 2000
image_size = 64
# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create the dataset
train_dataset = datasets.ImageFolder(
    root="../data/celeba-dataset/img_align_celeba",
    transform=transform
)

# Create the data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Create model, loss function and optimizer
model = VAE(channels=3, embedding_dim=200)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

class VAELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(VAELoss, self).__init__()

    def forward(self, inputs, targets, z_mean, z_log_var):    
        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt                     
        kl_divergence = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        mse = r_loss_factor * F.mse_loss(targets, inputs)
        vae_loss = mse + kl_divergence
        
        return vae_loss
    
criterion = VAELoss()

model.to(device)
criterion.to(device)
# print(summary(model, (3, image_size, image_size)))

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        # Forward pass
        reconstructed, z_mean, z_log_var = model(data)
        loss = criterion(reconstructed, data, z_mean, z_log_var)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, batch_idx+1, len(train_loader), loss.item()))

    print("\nEpoch [{}/{}], Loss: {:.4f}\n".format(epoch+1, num_epochs, total_loss/len(train_loader)))

# Save the trained model
torch.save(model, "vae_celeba.pth")
