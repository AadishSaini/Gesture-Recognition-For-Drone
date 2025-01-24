import cv2 as cv
import mediapipe as mp
import time
import os
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv.VideoCapture(0)

# Input gesture name and set up directories
name = input("Enter gesture name: ")
base_dir = 'dataset'

gesture_dir = os.path.join(base_dir, name)
frame_dir = os.path.join(gesture_dir, 'frame')
raw_dir = os.path.join(gesture_dir, 'raw')

# Create directories if they don't exist
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)

num = 0
last_capture_time = time.time()  # Initialize last capture time
capture_interval = 0.5  # Minimum interval (in seconds) between saving images

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("-> Frame not captured from the camera.\n")
        break

    # Process the frame for hand landmarks
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and bounding box on the original frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Create a black canvas for the raw hand wireframe
            raw_canvas = np.zeros_like(frame)

            # Draw landmarks on the black canvas
            mp_drawing.draw_landmarks(raw_canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the bounding box around the hand
            img_height, img_width, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img_width)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img_width)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img_height)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img_height)

            # Add padding to the bounding box
            padding = 20
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, img_width)
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, img_height)

            # Crop the hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]
            raw_crop = raw_canvas[y_min:y_max, x_min:x_max]

            # Save images at the defined interval
            current_time = time.time()
            if current_time - last_capture_time > capture_interval:
                # Save raw wireframe image
                raw_image_file = os.path.join(raw_dir, f"raw_{num}.png")
                cv.imwrite(raw_image_file, raw_crop)
                print(f"Saved raw wireframe image: {raw_image_file}")

                # Save annotated image
                annotated_image_file = os.path.join(frame_dir, f"frame_{num}.png")
                cv.imwrite(annotated_image_file, hand_crop)
                print(f"Saved annotated image: {annotated_image_file}")

                num += 1
                last_capture_time = current_time  # Update the last capture time

            # Draw the bounding box on the original frame (for visualization)
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Flip and resize the frame for display
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (1200, 800))
    cv.imshow('feed', frame)

    # Break the loop on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("-> Terminating.\n")
        break

# Release the webcam and close the window
cap.release()
cv.destroyAllWindows()
