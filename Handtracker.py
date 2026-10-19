import mediapipe as mp
import cv2
import numpy as np
import math

class Handtracker:
    def __init__(self, pinch_threshhold = 35):
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

        # State tracking variables
        self.drawn_square = None
        self.is_locked = False

        self.FINGER_TIPS = [8, 12, 16, 20]
        self.THUMBS = 4
        self.PINCH_THRESHOLD = pinch_threshhold


    def frame_process(self, frame):
        # hey mediapipe can't read opencv bgr format, converting it to rgb
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        left_thumb_pt = None   # Bottom-Left Anchor
        right_index_pt = None  # Top-Right Anchor

        left_pinched = False
        right_pinched = False

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                label = handedness.classification[0].label
                landmarks = hand_landmarks.landmark

                # Convert normalized coordinates to pixel values
                index_pt = (int(landmarks[self.FINGER_TIPS[0]].x * w), int(landmarks[self.FINGER_TIPS[0]].y * h))
                thumb_pt = (int(landmarks[self.THUMBS].x * w), int(landmarks[self.THUMBS].y * h))

                # Calculate distance between index tip and thumb tip for this hand
                pinch_dist = math.hypot(index_pt[0] - thumb_pt[0], index_pt[1] - thumb_pt[1])

                if label == "Left":
                    left_thumb_pt = thumb_pt 
                    if pinch_dist < self.PINCH_THRESHOLD:
                        left_pinched = True

                elif label == "Right":
                    right_index_pt = index_pt
                    if pinch_dist < self.PINCH_THRESHOLD:
                        right_pinched = True
            
        # --- UPDATE SQUARE GEOMETRY WHILE PINCHING ---
        if left_pinched and right_pinched and left_thumb_pt and right_index_pt:
            # Anchor 1: Bottom-Left (X1, Y2)
            x1, y2 = left_thumb_pt
            # Anchor 2: Top-Right (X2, Y1)
            x2, y1 = right_index_pt

            # Determine side length for a perfect square
            side = max(abs(x2 - x1), abs(y2 - y1))

            # Define 4 perfect square vertices (Clockwise)
            top_left = (x1, y1)
            top_right = (x2, y1)
            bottom_right = (x2, y2)
            bottom_left = (x1, y2)

            # Format for OpenCV polygon rendering
            pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
            self.drawn_square = pts.reshape((-1, 1, 2))
            self.is_locked = True

        if self.drawn_square is not None:
            cv2.polylines(frame, [self.drawn_square], isClosed=True, color=(0, 255, 0), thickness=3)

            status = "SQUARE UPDATING..." if (left_pinched and right_pinched) else "SQUARE LOCKED"
            cv2.putText(frame, status, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            status_text = f"Pinch Both Hands! (Left: {'YES' if left_pinched else 'NO'} | Right: {'YES' if right_pinched else 'NO'})"
            cv2.putText(frame, status_text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def reset_rectangle(self):
        self.drawn_square = None
        self.is_locked = False

    def capture(self, clean_frame):
        """Crop the drawn square out of a clean (undrawn) frame.
        Returns ((x_min, y_min, x_max, y_max), cropped_image) or (None, None)."""
        if self.drawn_square is None:
            return None, None

        pts = self.drawn_square.reshape(-1, 2)
        h, w = clean_frame.shape[:2]

        # min/max so it works regardless of which corner is which, clamped to the frame
        x_min = max(int(pts[:, 0].min()), 0)
        y_min = max(int(pts[:, 1].min()), 0)
        x_max = min(int(pts[:, 0].max()), w)
        y_max = min(int(pts[:, 1].max()), h)

        if x_max - x_min < 2 or y_max - y_min < 2:
            return None, None

        crop = clean_frame[y_min:y_max, x_min:x_max].copy()
        return (x_min, y_min, x_max, y_max), crop