import cv2

# Set the screen resolution
screen_width, screen_height = 1920, 1080  # You can adjust this based on your requirements

# Set the region to capture (you can adjust this based on your requirements)
capture_region = (0, 0, int(screen_width/2), screen_height)

# Open a connection to the screen capture
cap = cv2.VideoCapture(capture_region)

# Create a window to display the screen capture
cv2.namedWindow('Screen Capture', cv2.WINDOW_NORMAL)

while True:
    # Capture the screen image
    ret, frame = cap.read()

    # Display the captured image
    cv2.imshow('Screen Capture', frame)

    # Press 's' to capture a screenshot
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        # Save the screenshot
        cv2.imwrite('screenshot.png', frame)
        print('Screenshot saved!')

    # Break the loop when the user presses 'q'
    elif key & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
