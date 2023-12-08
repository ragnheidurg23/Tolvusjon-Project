import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import pickle


res_dir = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\ResNetRes\Results_4"
with open(os.path.join(res_dir, 'multiple_variables.pkl'), 'rb') as file:
    loaded_variables = pickle.load(file)

model_type = loaded_variables['transform']
transform = loaded_variables['transform']
dataset_directory_path = loaded_variables['dataset_directory_path']
test_image_paths = loaded_variables['test_image_paths']
model = loaded_variables['model']
if model_type is None or model_type == 'ResNet':
    model.fc = loaded_variables['model.fc']
elif model_type == 'VGG16':
    model.classifier[-1] = loaded_variables['model.classifier[-1]']
# model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
dataset = ImageFolder(root=dataset_directory_path, transform=transform)
num_classes = len(dataset.classes)

model.load_state_dict(torch.load(os.path.join(res_dir, 'model_weights.pth')))
model.eval()


# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    return input_batch


# Classify a single image
def classify_image(image_path, model):

    input_batch = preprocess_image(image_path).cuda()
    with torch.no_grad():

        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item()

class_names = dataset.classes  # Replace with your actual class names
# Test the image classification
for image_path in test_image_paths:
    predicted_class, confidence = classify_image(image_path, model)
    # Display the result
    class_actual = os.path.basename(os.path.dirname(image_path))

    print(f"Predicted Class: {class_names[predicted_class]}, Confidence: {confidence:.2%}, Actual: {class_actual}")
