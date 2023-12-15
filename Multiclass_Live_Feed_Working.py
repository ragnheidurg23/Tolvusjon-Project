import cv2
import torch
from torchvision import transforms
from torchvision import models

# Instantiate the model
model = models.efficientnet_b0(pretrained=True)

# Load the state dictionary
state_dict = torch.load('Models/Results_EfficientNetB0_dataset_all_all/model_weights.pth', map_location=torch.device('cpu'))

# Load the state dictionary into the model
model.load_state_dict(state_dict)

model.eval()

# Open a connection to the screen capture
cap = cv2.VideoCapture(1)

# Create a named window
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

# Set the window size
window_width = 640
window_height = 480
cv2.resizeWindow("Live Feed", window_width, window_height)

# Define a dictionary mapping class indices to names
# Update this dictionary according to your model's classes
class_names = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
    # Add other classes as necessary
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and resize
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (224, 224))

    # Transform frame to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(resized_frame)
    input_tensor = input_tensor.unsqueeze(0)

    # Model inference
    with torch.no_grad():
        predictions = model(input_tensor)

    # Get the predicted class index
    predicted_class = torch.argmax(predictions).item()

    # Map the predicted class to its name
    predicted_class_name = class_names.get(predicted_class, "Unknown")

    # Add class text to the frame
    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()