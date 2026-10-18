import cv2
import numpy as np
import mediapipe as mp
from Handtracker import Handtracker

# initialize the webcam & handtracker
cap = cv2.VideoCapture(0)
tracker = Handtracker()

if not cap.isOpened():
    print("The camera wasn't opened")
    exit()

print("Camera is opened! Press q to close")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("camera failed")
        break

    frame = cv2.flip(frame, 1)

    frame, total_fingers = tracker.frame_process(frame) 

    # == OPENCV ==
    # displaying the semi-transparent background for the text
    cv2.rectangle(frame, (20, 20), (320, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        f"Finger: {total_fingers}",
        (35, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    # Show camera ui
    cv2.imshow('Webcam feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()