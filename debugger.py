# FOR DEBUGGING, RUNNING ONLY handtracker.py

import cv2
import numpy as np
import mediapipe as mp
from handtracker import handtracker

# initialize the webcam & handtracker
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
tracker = handtracker()

# initialize capture count
capture_count = 0

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
    clean_frame = frame.copy()

    frame = tracker.frame_process(frame) 

    # == OPENCV ==
    # Show camera ui

    key = cv2.waitKey(1) & 0xFF
    cv2.imshow('Webcam feed', frame)


    if key == ord('s'): # capture
        coords, crop = tracker.capture(clean_frame)
        if crop is not None:
            capture_count += 1
            x1, y1, x2, y2 = coords
            print(f"Capture #{capture_count}: top-left=({x1}, {y1}), bottom-right=({x2}, {y2}), size={x2-x1}x{y2-y1}")
            cv2.imshow(f'Capture {capture_count}', crop)
        else:
            print("No square drawn yet")
    if key == ord('c'):
        tracker.reset_rectangle()
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()