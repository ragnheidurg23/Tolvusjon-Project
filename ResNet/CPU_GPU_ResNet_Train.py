
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Step 1: Prepare Your Dataset

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# Load your dataset
dataset = ImageFolder(root='/Users/magnusbenediktmagnusson/Documents/Computer Vision/dataset', transform=transform)

# Split into train and validation sets (optional)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

# Step 2: Initialize the ResNet Model

# Setup device - Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the ResNet Model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_classes = len(dataset.classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)  # Move the modified model to the device

# Step 3: Define Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: Train the Model


# [Previous parts of the script including imports, dataset preparation, and model setup...]

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

num_epochs = 10  # Set the number of epochs

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels)
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} [Loss: {running_loss / (i+1):.4f}, Accuracy: {running_accuracy / (i+1):.2f}%]')

    # Validation Phase
    
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Validation Accuracy: %d %%' % (100 * correct / total))


    # Validation Loop
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total

# Modifying the training loop to include tqdm progress bar and enhanced metric display
for epoch in range(num_epochs):  # Assuming 'num_epochs' is defined in the original script
    running_loss = 0.0
    running_accuracy = 0.0
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (inputs, labels) in pbar:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, labels)
        pbar.set_description(f'Epoch {epoch + 1}/{num_epochs} [Loss: {running_loss / (i+1):.4f}, Accuracy: {running_accuracy / (i+1):.2f}%]')
