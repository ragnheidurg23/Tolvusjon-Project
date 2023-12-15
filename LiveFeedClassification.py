import cv2
import pyautogui
import numpy as np

# Set the screen resolution
screen_width, screen_height = pyautogui.size()

# Set the region to capture (you can adjust this based on your requirements)
capture_region = (0, 0, int(screen_width/2), screen_height)

# Create a window to display the screen capture
cv2.namedWindow('Screen Capture', cv2.WINDOW_NORMAL)

while True:
    # Capture the screen image
    screenshot = pyautogui.screenshot(region=capture_region)

    # Convert the PIL Image to a NumPy array
    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    # Display the captured image
    cv2.imshow('Screen Capture', frame)

    # Break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the window and close it
cv2.destroyAllWindows()
