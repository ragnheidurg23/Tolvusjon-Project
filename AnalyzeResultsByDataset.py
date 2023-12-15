import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.datasets import ImageFolder
import pickle
import numpy as np


results_path = r"C:\Users\arnar\PycharmProjects\CompVis\CustomObjectDetector\results\AllDatasetClassificationResults"
with open(os.path.join(results_path, 'models_results.pkl'), 'rb') as file:
    loaded_variables = pickle.load(file)


num_classes = loaded_variables['AlexNet']['num_classes']
throw_out_threshold = 50
count = 0
right1 = 0
wrong1 = 0
right2 = 0
wrong2 = 0
right3 = 0
wrong3 = 0
output_file_txt_str = ""
models_rejected = set()
for i in range(len(loaded_variables['AlexNet']['images'])):
    image = loaded_variables['AlexNet']['images'][i]
    class_actual = loaded_variables['AlexNet']['class'][i]

    # Score 1 is just tally up votes for each and winner wins
    score_1 = np.array([0 for j in range(num_classes)], dtype=float)
    # Score 2 we use the confidence of each as a weight and count that
    score_2 = np.array([0 for j in range(num_classes)], dtype=float)
    # Score 3 also takes into account the model accuracy
    score_3 = np.array([0 for j in range(num_classes)], dtype=float)
    score_continuous = []
    for model_type in loaded_variables:
        if model_type == "test_image_paths":
            continue
        if loaded_variables[model_type]['accuracy'] < throw_out_threshold:
            models_rejected.add(model_type)
            continue
        probabilities = loaded_variables[model_type]['probabilities'][i]

        # Calculate the weighted sum of indices
        weighted_sum_indices = np.sum(np.arange(len(probabilities)) * probabilities)
        print(weighted_sum_indices)
        # Calculate the sum of values
        sum_values = np.sum(probabilities)
        # Calculate the average index
        average_index = weighted_sum_indices / sum_values
        score_continuous.append(average_index)
        score_1[np.argmax(probabilities)] += 1
        score_2 += probabilities
        score_3 += loaded_variables[model_type]['accuracy']/100 * probabilities
    print("_________________________________")
    print("Image:", image)
    print("Actual Class", class_actual)
    print("Continuous Score: ", score_continuous)
    print("Average Continuous Score", sum(score_continuous)/len(score_continuous))
    print("Score 1 Results: ", score_1)
    print("Score 2 Results: ", score_2)
    print("Score 3 Results: ", score_3)
    output_file_txt_str += "_________________________________" + "\n"
    output_file_txt_str += "Image: " + image + "\n"
    output_file_txt_str += "Actual Class:\t" + class_actual + "\n"
    output_file_txt_str += "Continuous Score: " + str(score_continuous) + "\n"
    output_file_txt_str += "Average Continuous Score:\t" + str(sum(score_continuous)/len(score_continuous)) + "\n"
    output_file_txt_str += "Score 1 Results: " + str(score_1) + "\n"
    output_file_txt_str += "Score 2 Results: " + str(score_2) + "\n"
    output_file_txt_str += "Score 3 Results: " + str(score_3) + "\n"

    count += 1
    if num_classes == 4:
        if int(class_actual.split('_')[1]) == np.argmax(score_1):
            right1 += 1
        if int(class_actual.split('_')[1]) == np.argmax(score_2):
            right2 += 1
        if int(class_actual.split('_')[1]) == np.argmax(score_3):
            right3 += 1
    else:
        if int(class_actual.split('_')[1]) == np.argmax(score_1):
            right1 += 1
        elif int(class_actual.split('_')[1]) == 3 and 1 == np.argmax(score_1):
            right1 += 1
        else:
            wrong1 += 1
        if int(class_actual.split('_')[1]) == np.argmax(score_2):
            right2 += 1
        elif int(class_actual.split('_')[1]) == 3 and 1 == np.argmax(score_2):
            right2 += 1
        else:
            wrong2 += 1
        if int(class_actual.split('_')[1]) == np.argmax(score_3):
            right3 += 1
        elif int(class_actual.split('_')[1]) == 3 and 1 == np.argmax(score_3):
            right3 += 1
        else:
            wrong3 += 1

output_file_txt_str += "\n\nModel CutOff Threshold: " + str(throw_out_threshold) + "\n"
output_file_txt_str += "Models Rejected:\n\t" + "\n\t".join(models_rejected) + "\n"

print("Accuracy score 1: ", 100*right1/count)
print("Accuracy score 2: ", 100*right2/count)
print("Accuracy score 3: ", 100*right3/count)
output_file_txt_str += "Accuracy score 1: " + str(100*right1/count) + "\n"
output_file_txt_str += "Accuracy score 2: " + str(100*right2/count) + "\n"
output_file_txt_str += "Accuracy score 3: " + str(100*right3/count) + "\n"

with open(os.path.join(results_path, "ResultsTxt.txt"), "w+") as file:
    file.write(output_file_txt_str)