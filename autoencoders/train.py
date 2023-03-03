import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Autoencoder
from torchsummary import summary
# Define hyperparameters
batch_size = 64
learning_rate = 1e-3
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
model = Autoencoder(channels=1, embedding_dim=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.to(device)
criterion.to(device)

# print(summary(model, (1,32,32)))

# Train the model
for epoch in range(num_epochs):
    model.train()
    for img, _ in train_loader:
        img = img.to(device)
        recon = model(img)
        loss = criterion(recon, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model, "autoencoder.pth")
