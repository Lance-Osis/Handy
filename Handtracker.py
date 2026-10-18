import mediapipe as mp
import cv2

class Handtracker:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cam = cv2.flip(frame, 1)

    def __init__(self):
        # Intialize Mediapipe
        self.mp_hands = mp.solutions.hands  
        self.mp_draw = mp.solutions.drawing_utils
        # Intialize hand tracker
        self.hands = self.mp_hands.Hands (
        static_image_mode = False,
        max_num_hands = 2,
        min_detection_confidence = 0.8,
        min_tracking_confidence = 0.8
        )

        self.FINGER_TIPS = [8, 12, 16, 20]
        self.THUMBS = 4

    def frame_process(self, frame):
        # hey mediapipe can't read opencv bgr format, converting it to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        total_fingers = 0

        if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # this is the process for drawing the skeletal landmark
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        
                    # is it left or right
                    hand_label = handedness.classification[0].label
        
                    landmark = hand_landmarks.landmark
                    fingers_up = []
        
                    # -- check thumbs --
                    # based on the horizotal position on the thumb this algorithm checks whether your shi is left or right
                    if hand_label == "Right":
                        if landmark[self.THUMBS].x < landmark[self.THUMBS - 1].x:
                            fingers_up.append(1)
                            print("right hand")
                        else:
                            fingers_up.append(0)
                
                    # for left hand
                    else:
                        if landmark[self.THUMBS].x > landmark[self.THUMBS - 1].x:
                            fingers_up.append(1)
                            print("left hand")
                        else:
                            fingers_up.append(0)
        
                    # -- check four fingers --
                    for tip in self.FINGER_TIPS:
                        if landmark[tip].y < landmark[tip-2].y:
                            fingers_up.append(1)
                        else:
                            fingers_up.append(0)
        
                    total_fingers += sum(fingers_up)
        return frame, total_fingers