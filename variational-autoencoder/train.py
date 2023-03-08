import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import VAE
from torchsummary import summary

# Define hyperparameters
batch_size = 64
learning_rate = 0.0005
num_epochs = 10

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

model.to(device)

# print(summary(model, (1,32,32)))

# Train the model for 10 epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, _) in enumerate(train_loader):
        # Move the input data to the device
        inputs = inputs.to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Feed the input data to the model to get the output
        z_mean, z_log_var, reconstruction = model(inputs)

        # Calculate the loss using the output and the ground truth labels
        loss = model.loss_function(inputs, z_mean, z_log_var, reconstruction)

        # Backpropagate the loss to update the model parameters
        loss.backward()
        optimizer.step()

        # Print the loss after every 100 batches
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0