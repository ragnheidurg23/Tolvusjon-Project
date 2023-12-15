import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import shutil
import os
import pickle
import random


def save_model(model_):
    path = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\results2"
    if not os.path.exists(path):
        os.mkdir(path)
    if CUSTOM_DATASET_ON:
        unique_dir_name = model_type + "_" + os.path.basename(dataset_directory_path) + "Custom"
    else:
        unique_dir_name = model_type + "_" + os.path.basename(dataset_directory_path)
    new_res_dir_name = os.path.join(path, "Results_" + unique_dir_name)
    if not os.path.exists(new_res_dir_name):
        os.mkdir(new_res_dir_name)
    torch.save(model_.state_dict(), os.path.join(new_res_dir_name, 'model_weights.pth'))
    return new_res_dir_name


# TODO: Try ObjectNet, bigger resnet, AlexNet
# TODO: Check out learning rate, Transformers
# TODO: Get pretrained ViT (vision transformer) and BiT (Big Transfer)?
# TODO: Get pretrained transformer, they are not necessarily included in models
# TODO: Check out few shot learning models!
weights, model = None, None
TRAIN = True
num_epochs = 20  # Set the number of epochs
model_types = [
    'VGG16',
    'ResNet50',
    'ResNet101',
    'ViTB16',
    "ViTB32",
    "AlexNet",
    'DenseNet121',
    'EfficientNetB0',
    'GoogleNet',
]
datasets = [
    'dataset_0_3_all',
    'dataset_all_all',
    'dataset_MGD1K_0123',
    'dataset_all_allCustom',
    # 'dataset_ricc_train_0123'
]
CUSTOM_DATASET_ON = False
CUSTOM_DATASET = {
    0: {
        'train': 300,
        'test': 60,
        'valid': 60
    },
    1: {
        'train': 300,
        'test': 233,
        'valid': 233
    },
    2: {
        'train': 300,
        'test': 14,
        'valid': 13
    },
    3: {
        'train': 220,
        'test': 10,
        'valid': 15
    }
}
dataset_directory_path_root = r'C:\Users\arnar\PycharmProjects\CompVis\DryEyes'

res_dir = ""

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


for data_set_name in datasets:
    if data_set_name == 'dataset_all_allCustom':
        CUSTOM_DATASET_ON = True
        data_set_name = "dataset_all_all"
    else:
        CUSTOM_DATASET_ON = False
    dataset_directory_path = os.path.join(dataset_directory_path_root, data_set_name)
    # Define your transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ])
    dataset = ImageFolderWithPaths(root=dataset_directory_path, transform=transform)
    # Split into train and validation sets (optional)
    if CUSTOM_DATASET_ON:
        dataset = ImageFolderWithPaths(root=r"C:\Users\arnar\PycharmProjects\CompVis\DryEyes\dataset_all_all", transform=transform)

        # Initialize lists to store the indices for each split
        train_indices, valid_indices, test_indices = [], [], []

        # Iterate over each class
        for class_label in range(len(dataset.classes)):
            # Get all indices corresponding to the current class
            class_indices = [i for i, target in enumerate(dataset.targets) if target == class_label]

            # Randomly shuffle the indices
            random.shuffle(class_indices)

            # Extract the desired number of indices for each split
            train_size = CUSTOM_DATASET[class_label]['train']
            valid_size = CUSTOM_DATASET[class_label]['valid']
            test_size = CUSTOM_DATASET[class_label]['test']

            train_indices += class_indices[:train_size]
            valid_indices += class_indices[train_size:train_size + valid_size]
            test_indices += class_indices[train_size + valid_size:train_size + valid_size + test_size]

        # Create Subset datasets for training, validation, and testing
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, valid_indices)
        test_set = Subset(dataset, test_indices)

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        test_loader = DataLoader(test_set, batch_size=32)
        test_image_paths = [item[2] for item in test_set]

    elif data_set_name != 'dataset_ricc_train_0123':
        train_size = int(0.7 * len(dataset))
        val_size = int((len(dataset) - train_size) / 2)
        test_size = len(dataset) - val_size - train_size
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        test_loader = DataLoader(test_set, batch_size=32)
        test_image_paths = [item[2] for item in test_set]
    else:
        train_size = int(0.85 * len(dataset))
        val_size = int(len(dataset) - train_size)
        # test_size = len(dataset) - val_size - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        dataset_test = ImageFolderWithPaths(root=r"C:\Users\arnar\PycharmProjects\CompVis\DryEyes\rick_train", transform=transform)
        test_set = random_split(dataset_test, [len(dataset_test)])

        # Create data loaders
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=32)
        test_loader = DataLoader(dataset_test, batch_size=32)
        test_image_paths = [item[2] for item in test_set]

    for model_type in model_types:
        # model_type = 'VGG16'
        if model_type == 'VGG16':
            from torchvision.models import VGG16_Weights
            weights = VGG16_Weights
            model = models.vgg16(weights=weights)
        elif model_type == 'ResNet50':
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1
            models.resnet50(weights=weights)
        elif model_type == 'ResNet101':
            from torchvision.models import ResNet101_Weights
            weights = ResNet101_Weights
            model = models.resnet101(weights=weights)
        elif model_type == 'ViTB16':
            from torchvision.models import ViT_B_16_Weights
            weights = ViT_B_16_Weights
            model = models.vit_b_16(weights=weights)
        elif model_type == "ViTB32":
            from torchvision.models import ViT_B_32_Weights
            weights = ViT_B_32_Weights
            model = models.vit_b_32(weights=weights)
        elif model_type == "ViTL16":
            from torchvision.models import ViT_L_16_Weights
            weights = ViT_L_16_Weights
            model = models.vit_l_16(weights=weights)
        elif model_type == 'ViTL32':
            from torchvision.models import ViT_L_32_Weights
            weights = ViT_L_32_Weights
            model = models.vit_l_32(weights=weights)
        elif model_type == 'AlexNet':
            # TODO TRY
            from torchvision.models import AlexNet_Weights
            model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        elif model_type == "DenseNet121":
            from torchvision.models import DenseNet121_Weights
            weights = DenseNet121_Weights
            model = models.densenet121(weights=weights)
        elif model_type == "EfficientNetB0":
            from torchvision.models import EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights
            model = models.efficientnet_b0(weights=weights)
        elif model_type == "GoogleNet":
            from torchvision.models import GoogLeNet_Weights
            weights = GoogLeNet_Weights
            model = models.googlenet(weights=weights)
        elif model_type == "InceptionV3":
            from torchvision.models import Inception_V3_Weights
            weights = Inception_V3_Weights
            model = models.inception_v3(weights=weights)


        data_save_dict = {}
        data_save_dict['model_type'] = model_type
        data_save_dict['num_epochs'] = num_epochs
        data_save_dict['transform'] = transform
        if CUSTOM_DATASET_ON:
            data_save_dict['dataset_directory_path'] = dataset_directory_path+"Custom"
        else:
            data_save_dict['dataset_directory_path'] = dataset_directory_path
        data_save_dict['test_image_paths'] = test_image_paths
        data_save_dict['weights'] = weights
        data_save_dict['model'] = model
        # Setup device - Use GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        # Training Loop
        accuracy_to_beat = 0
        model_saved_early = False
        model_saved_at_epoch = 0
        running_losses = []
        running_accuracies = []
        validation_accuracies = []
        if TRAIN:
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                running_accuracy = 0.0
                running_accuracies.append([])
                running_losses.append([])
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
                    running_losses[-1].append(running_loss)
                    running_accuracies[-1].append(running_accuracy)
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
                validation_accuracies.append(100 * correct / total)
                if epoch > 14 and (100 * correct / total) >= accuracy_to_beat:
                    res_dir = save_model(model)
                    model_saved_early = True
                    accuracy_to_beat = (100 * correct / total)
                    model_saved_at_epoch = epoch
                # Validation Loop
                model.eval()
                with torch.no_grad():
                    for inputs, labels, _ in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
        data_save_dict['running_losses'] = running_losses
        data_save_dict['running_accuracies'] = running_accuracies
        data_save_dict['validation_accuracies'] = validation_accuracies


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
            if not model_saved_early:
                res_dir = save_model(model)
            copy_test_set(test_image_paths, res_dir)
            data_save_dict['res_dir'] = res_dir
        else:
            res_dir = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\ResNetRes\Results_VGG16_dataset_ricc_train_0123"
        with open(os.path.join(res_dir, 'multiple_variables.pkl'), 'wb') as file:
            pickle.dump(data_save_dict, file)

