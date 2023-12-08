import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG16_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import shutil
import os
import pickle

model_type = 'VGG16'
weights = VGG16_Weights # ResNet50_Weights.IMAGENET1K_V1
model = models.vgg16(weights=weights)  # models.resnet50(weights=weights)
dataset_directory_path = r'C:\Users\arnar\PycharmProjects\CompVis\DryEyes\dataset'

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

data_save_dict = {}
data_save_dict['model_type'] = model_type
TRAIN = True

# Step 1: Prepare Your Dataset

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size of ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])
data_save_dict['transform'] = transform
# Load your dataset

dataset = ImageFolderWithPaths(root=dataset_directory_path, transform=transform)
data_save_dict['dataset_directory_path'] = dataset_directory_path

# Split into train and validation sets (optional)
train_size = int(0.7 * len(dataset))
val_size = int((len(dataset) - train_size)/2)
test_size = len(dataset) - val_size - train_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(val_set, batch_size=32)

test_image_paths = [item[2] for item in test_set]
data_save_dict['test_image_paths'] = test_image_paths
# Step 2: Initialize the ResNet Model

# Setup device - Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the ResNet Model

data_save_dict['weights'] = weights

data_save_dict['model'] = model
num_classes = len(dataset.classes)
if model_type == 'VGG16':
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    data_save_dict['model.classifier[-1]'] = model.classifier[-1]
elif model_type == 'ResNet':
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    data_save_dict['model.fc'] = model.fc
model = model.to(device)  # Move the modified model to the device

# Step 3: Define Loss Function and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total


num_epochs = 10  # Set the number of epochs

# Training Loop
if TRAIN:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (inputs, labels, _) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels)
            pbar.set_description(
                f'Epoch {epoch + 1}/{num_epochs} [Loss: {running_loss / (i + 1):.4f}, Accuracy: {running_accuracy / (i + 1):.2f}%]')

        # Validation Phase

        model.eval()  # Set the model to evaluation mode
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Validation Accuracy: %d %%' % (100 * correct / total))

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

def save_model(model_):
    path = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\ResNetRes"
    new_res_dir_name = os.path.join(path, "Results_"+str(len(os.listdir(path))+1))
    os.mkdir(new_res_dir_name)
    torch.save(model_.state_dict(), os.path.join(new_res_dir_name, 'model_weights.pth'))
    return new_res_dir_name

def copy_test_set(test_image_paths_, res_dir_):
    # Access image paths in the test set
    if not os.path.exists(os.path.join(res_dir_, 'test')):
        os.mkdir(os.path.join(res_dir_, 'test'))
    for image_path in test_image_paths_:
        basename = os.path.basename(os.path.dirname(image_path))
        if not os.path.exists(os.path.join(res_dir_,'test', basename)):
            os.mkdir(os.path.join(res_dir_,'test', basename))
        shutil.copy(image_path, os.path.join(res_dir_,'test', basename))


if TRAIN:
    res_dir = save_model(model)
    copy_test_set(test_image_paths, res_dir)
    data_save_dict['res_dir'] = res_dir
else:
    res_dir = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\ResNetRes\Results_3"
with open(os.path.join(res_dir, 'multiple_variables.pkl'), 'wb') as file:
    pickle.dump(data_save_dict, file)
