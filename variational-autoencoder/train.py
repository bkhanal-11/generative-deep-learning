import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import VAE
from torchsummary import summary

# Define hyperparameters
batch_size = 16
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
criterion = nn.MSELoss()

model.to(device)
criterion.to(device)
# print(summary(model, (1,32,32)))
# print(next(iter(i.size() for i,j in train_loader)))

# Train the model for 10 epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, _) in enumerate(train_loader):
        # Move the input data to the device
        inputs = inputs.to(device)

        # Feed the input data to the model to get the output
        reconstructed = model(inputs)
        # Clear the gradients
        optimizer.zero_grad()

        # Calculate the loss using the output and the ground truth labels
        # loss = model.loss_function(inputs, z_mean, z_log_var, reconstruction)

        loss_mse = criterion(inputs, reconstructed)
        loss = loss_mse + model.encoder.kl_div

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
