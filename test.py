from tensorflow.keras.models import load_model
import cv2 as cv
import numpy as np
import mediapipe as mp


#because i made a heavenly mistake while triaing :pray:
gesture_mapping = {0: 'backward', 1: 'down', 2: 'down_for_right_hand', 3: 'forward', 4: 'left', 5: 'left_for_right_hand', 6: 'right', 7: 'right_for_right_hand', 8: 'up', 9: 'up_for_right_hand'}


model = load_model('final_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)
image_size = 128

while cap.isOpened():
    ret, frame = cap.read()
    showing = frame.copy()
    frame = cv.flip(frame, 1)
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label 

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

            if hand_crop.size > 0:
                hand_crop = cv.cvtColor(hand_crop, cv.COLOR_BGR2GRAY)  
                hand_crop = cv.resize(hand_crop, (image_size, image_size))  
                hand_crop = np.expand_dims(hand_crop, axis=-1)  
                hand_crop = np.expand_dims(hand_crop, axis=0)  
                hand_crop = hand_crop / 255.0 

                predictions = model.predict(hand_crop, verbose=0)
                gesture_idx = np.argmax(predictions)
                confidence = np.max(predictions)
                gesture_name = gesture_mapping.get(gesture_idx, "Unknown")

                # for shortening the gesture name for right hands
                if gesture_name == "right_for_right_hand":
                    gesture_name = "right"
                if gesture_name == "left_for_right_hand":
                    gesture_name = "left"
                if gesture_name == "up_for_right_hand":
                    gesture_name = "up"
                if gesture_name == "down_for_right_hand":
                    gesture_name = "down"
                label_text = f"{hand_label} Hand: {gesture_name} ({confidence:.2f})"
                cv.putText(frame, label_text,
                           (x_min, y_min - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    frame = cv.resize(frame, (1200, 800))  
    cv.imshow("Hand Gesture Recognition", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
