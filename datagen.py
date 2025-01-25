import cv2 as cv
import mediapipe as mp
import time
import os
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

name = input("Enter gesture name jo chahiye: ")
base_dir = 'dataset'

gesture_dir = os.path.join(base_dir, name)
frame_dir = os.path.join(gesture_dir, 'frame')
raw_dir = os.path.join(gesture_dir, 'raw')

os.makedirs(frame_dir, exist_ok=True)
os.makedirs(raw_dir, exist_ok=True)

num = 0
last_capture_time = time.time()  
interval = 0.5  

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("-> Frame not captured from the camera.\n")
        break

    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            khali = np.zeros_like(frame)
            mp_drawing.draw_landmarks(khali, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            #i gpt'ed this part uwu
            img_height, img_width, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * img_width)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * img_width)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * img_height)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * img_height)

            padding = 20
            x_min = max(x_min - padding, 0)
            x_max = min(x_max + padding, img_width)
            y_min = max(y_min - padding, 0)
            y_max = min(y_max + padding, img_height)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            raw_crop = khali[y_min:y_max, x_min:x_max]

            
            # click pictures 
            current_time = time.time()
            if current_time - last_capture_time > capture_interval:   #only when time is more than 0.5 seconds than the last time :pogchamp:
                raw_image_file = os.path.join(raw_dir, f"raw_{num}.png")
                cv.imwrite(raw_image_file, raw_crop)
                print(f"raw image file saved: {raw_image_file}")

                frame_image_file = os.path.join(frame_dir, f"frame_{num}.png")
                cv.imwrite(frame_image_file, hand_crop)
                print(f"frame file saved: {frame_image_file}")

                num += 1
                last_capture_time = current_time  

            #bound the hand
            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # flip and make the image bada
    frame = cv.flip(frame, 1)
    frame = cv.resize(frame, (1200, 800))
    cv.imshow('feed', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("-> Terminating.\n")
        break

cap.release()
cv.destroyAllWindows()
