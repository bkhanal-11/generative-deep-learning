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
num_epochs = 200
r_loss_factor = 500

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Create model, loss function and optimizer
model = VAE(channels=1, embedding_dim=2)

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
# print(summary(model, (1,32,32)))

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, _) in enumerate(train_loader):
        # Move the input data to the device
        inputs = inputs.to(device)

        # Feed the input data to the model to get the output
        reconstructed, z_mean, z_log_var = model(inputs)
        optimizer.zero_grad()

        # Calculate the loss using the output and the ground truth labels
        loss = criterion(reconstructed, inputs, z_mean, z_log_var)

        # Backpropagate the loss to update the model parameters
        loss.backward()
        optimizer.step()

        # Print the loss after every 100 batches
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Save the trained model
torch.save(model, "vae_mnist.pth")
