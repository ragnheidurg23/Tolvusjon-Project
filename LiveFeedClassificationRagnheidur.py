import cv2
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Load your custom EfficientNet model with PyTorch
model = EfficientNet.from_name('efficientnet-b0', num_classes=2)  # Specify num_classes
model.load_state_dict(torch.load('Models\Results_EfficientNetB0_dataset_0_3_all\model_weights.pth', ap_location=torch.device('cpu')))
model.eval()

# Set the screen resolution
# screen_width, screen_height = 1920, 1080  # You can adjust this based on your requirements

# # Set the region to capture (you can adjust this based on your requirements)
# capture_region = (0, 0, int(screen_width/2), screen_height)

# Open a connection to the screen capture
cap = cv2.VideoCapture(0)

# Create a window to display the screen capture
cv2.namedWindow('Screen Capture', cv2.WINDOW_NORMAL)

# Define a transform to preprocess the frame for the model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

while True:
    # Capture the screen image
    ret, frame = cap.read()

    # Preprocess the frame for the model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = transform(frame)
    frame = frame.unsqueeze(0)  # Add batch dimension

    # Predict the class probabilities
    with torch.no_grad():
        predictions = model(frame)

    # Get the predicted class index
    predicted_class = torch.argmax(predictions).item()

    # Display the predicted class on the frame
    cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the captured image
    cv2.imshow('Screen Capture', frame)

    # Press 's' to capture a screenshot
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        # Save the screenshot
        cv2.imwrite('screenshot.png', cv2.cvtColor(frame.squeeze().numpy().transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))
        print('Screenshot saved!')

    # Break the loop when the user presses 'q'
    elif key & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()