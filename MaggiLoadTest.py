import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import pickle


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

# Preprocess the input image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path)
    except AttributeError:
        image_path, _, _ = image_path
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
def classify_image(image_path, model, number_classes):
    input_batch = preprocess_image(image_path).cuda()
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Return all class probabilities
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[predicted_class].item(), probabilities.cpu().numpy()[:number_classes], input_batch


data_dict = {}
data_set_type_output_dir = "all_allCustom"
main_results_dir = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\results2"
output_save_text_file_str = ""
for res_dir in os.listdir(main_results_dir):
    print(output_save_text_file_str)
    predicted_classes = {}
    actual_classes = {}
    if data_set_type_output_dir in res_dir and "CUSTOM" not in res_dir:
        print(res_dir)
        res_dir = os.path.join(r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\results2", res_dir)
    else:
        continue
    model_name = os.path.basename(res_dir)
    with open(os.path.join(res_dir, 'multiple_variables.pkl'), 'rb') as file:
        loaded_variables = pickle.load(file)

    model_type = loaded_variables['model_type']
    transform = loaded_variables['transform']
    dataset_directory_path = loaded_variables['dataset_directory_path']
    test_image_paths = loaded_variables['test_image_paths']
    model = loaded_variables['model']
    if 'test_image_paths' not in data_dict:
        data_dict['test_image_paths'] = test_image_paths
    dataset = ImageFolder(root=dataset_directory_path, transform=transform)
    num_classes = len(dataset.classes)

    model.load_state_dict(torch.load(os.path.join(res_dir, 'model_weights.pth')))
    model.eval()

    class_names = dataset.classes  # Replace with your actual class names
    data_dict[model_type] = {}
    data_dict[model_type]['transform'] = transform
    data_dict[model_type]['num_classes'] = num_classes
    data_dict[model_type]['model'] = model

    # Test the image classification
    right, wrong, combined = 0, 0, 0
    TP = {}
    FP = {}
    TN = {}
    FN = {}
    for class_name in class_names:
        TP[class_name] = 0
        FP[class_name] = 0
        TN[class_name] = 0
        FN[class_name] = 0

    data_dict[model_type]['images'] = []
    data_dict[model_type]['probabilities'] = []
    data_dict[model_type]['class'] = []
    for image_path in test_image_paths:
        predicted_class, confidence, all_probabilities, input_tensor = classify_image(image_path, model, num_classes)
        if predicted_class not in predicted_classes:
            predicted_classes[predicted_class] = 0
        predicted_classes[predicted_class] += 1
        class_actual = os.path.basename(os.path.dirname(image_path))
        if class_actual not in actual_classes:
            actual_classes[class_actual] = 0
        actual_classes[class_actual] += 1
        target_class = int(class_actual.split("_")[1])

        data_dict[model_type]['images'].append(os.path.basename(image_path))
        data_dict[model_type]['probabilities'].append(all_probabilities)
        data_dict[model_type]['class'].append(class_actual)
        # Display the result
        if class_names[predicted_class] == class_actual:
            right += 1
        elif class_names[predicted_class] != class_actual:
            wrong += 1
        combined += 1
        if class_names[predicted_class] == class_actual:
            TP[class_actual] += 1
        elif class_names[predicted_class] != class_actual:
            FP[class_names[predicted_class]] += 1
            FN[class_actual] += 1

        print(f"Predicted Class: {class_names[predicted_class]}, Confidence: {confidence:.2%}, Actual: {class_actual}")
    predicted_classes = dict(sorted(predicted_classes.items()))
    actual_classes = dict(sorted(actual_classes.items()))
    output_save_text_file_str += "______________________________________\n"
    output_save_text_file_str += "Model:\t" + model_type + "\n"
    output_save_text_file_str += f"Correctly classified: {str(right)}\t|Incorrectly classified: {str(wrong)}\t|Accuracy Score: {str(right/combined*100)}" + "\n"
    output_save_text_file_str += f"Predicted Class distribution:   " + "\t".join([f"grade_{str(i)}: {predicted_classes[i]}" for i in predicted_classes]) + "\n"
    output_save_text_file_str += f"Test Dataset Class Distribution:" + "\t".join([f"{str(i)}: {actual_classes[i]}" for i in actual_classes]) + "\n"

    print("______________________________________")
    print("Model:\t", model_type)
    print(f"Correctly classified: {right}\t|Incorrectly classified: {wrong}\t|Accuracy Score: {right/combined*100}")
    print(f"Predicted Class distribution:   ", "\t".join([f"grade_{str(i)}: {predicted_classes[i]}" for i in predicted_classes]))
    print(f"Test Dataset Class Distribution:", "\t".join([f"{str(i)}: {actual_classes[i]}" for i in actual_classes]))
    data_dict[model_type]['right'] = right
    data_dict[model_type]['wrong'] = wrong
    data_dict[model_type]['accuracy'] = right/combined*100
    for class_name in class_names:

        try:
            precision = TP[class_name]/(TP[class_name]+FP[class_name])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = TP[class_name]/(TP[class_name]+FN[class_name])
        except ZeroDivisionError:
            recall = 0
        data_dict[model_type][class_name] = {}
        data_dict[model_type][class_name]['precision'] = precision
        data_dict[model_type][class_name]['recall'] = recall
        output_save_text_file_str += "Class: " + class_name + "\n"
        output_save_text_file_str += "Precision" + str(precision) + "\t" + "Recall" + str(recall) + "\n"
        print("Class: ", class_name)
        print("Precision", precision, "\t", "Recall", recall)

models_output_dir = ""
if data_set_type_output_dir == "_0_3_all":
    models_output_dir = os.path.join(main_results_dir, "BinaryClassificationResults")
elif data_set_type_output_dir == "MGD1K_0123":
    models_output_dir = os.path.join(main_results_dir, "MGD1kClassificationResults")
elif data_set_type_output_dir == "all_all":
    models_output_dir = os.path.join(main_results_dir, "AllDatasetClassificationResults")
elif data_set_type_output_dir == "all_allCustom":
    models_output_dir = os.path.join(main_results_dir, "AllCustomDatasetClassificationResults")
if not os.path.exists(models_output_dir):
    os.mkdir(models_output_dir)

with open(os.path.join(models_output_dir, 'models_results.pkl'), 'wb+') as file:
    pickle.dump(data_dict, file)
with open(os.path.join(models_output_dir, 'textOutput.txt'), 'w+') as file:
    file.write(output_save_text_file_str)
