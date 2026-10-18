import cv2
import numpy as np
import mediapipe as mp

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Intialize Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Intialize hand tracker
hands = mp_hands.Hands (
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
    )

FINGER_TIPS = [8, 12, 16, 20]
THUMBS = 4

if not cap.isOpened():
    print("The camera wasn't open")
    exit()

print("Camera is opened! Press q to close")

while cap.isOpened():
    ret, frame = cap.read()
    cam = cv2.flip(frame, 1)

    # == MEDIAPIPE ==
    total_fingers = 0

    # hey mediapipe can't read opencv bgr, converting it to rgb
    rgb_frame = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # this is the process for drawing the skeletal landmark
            mp_draw.draw_landmarks(
                cam,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # is it left or right
            hand_label = handedness.classification[0].label

            landmark = hand_landmarks.landmark
            fingers_up = []

            # -- check thumbs --
            # based on the horizotal position on the thumb this algorithm checks whether your shi is left or right
            if hand_label == "Right":
                if landmark[THUMBS].x < landmark[THUMBS - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)
        
            # for left hand
            else:
                if landmark[THUMBS].x > landmark[THUMBS - 1].x:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            # -- check four fingers --
            for tip in FINGER_TIPS:
                if landmark[tip].y < landmark[tip-2].y:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers += sum(fingers_up)

    # == OPENCV ==
    # displaying the semi-transparent background for the text
    cv2.rectangle(cam, (20, 20), (320, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        cam,
        f"Finger: {total_fingers}",
        (35, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    # Show camera ui
    cv2.imshow('Webcam feed', cam)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()