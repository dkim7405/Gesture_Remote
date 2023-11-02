import mediapipe as mp
import numpy as np
import cv2

class DetectHand:

    def __init__(self):
        self.num_hands = 1
        self.hand_Identify = mp.solutions.hands
        self.hand_detector_model = self.hand_Identify.Hands(max_num_hands=self.num_hands, min_detection_confidence=0.8, min_tracking_confidence=0.5)
        self.drawer = mp.solutions.drawing_utils

    def detect_from_frame(self, img):
        detect_img = img.copy()
        detect_img = cv2.flip(img, 1)
        detect_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detect_img.flags.writeable = False
        processed = self.hand_detector_model.process(detect_img)
        detect_img.flags.writeable = True

        return processed.multi_hand_landmarks
    
    def detect_from_frame_normalized(self, img):
        detect_img = img.copy()
        detect_img = cv2.flip(img, 1)
        detect_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        detect_img.flags.writeable = False
        processed = self.hand_detector_model.process(detect_img)
        detect_img.flags.writeable = True

        return processed.multi_hand_world_landmarks
    
    def processed_from_frame(self, img):
        result_img = img.copy()
        hand_landmarks = self.detect_from_frame(result_img)
        if hand_landmarks:
            for i in hand_landmarks:
                self.drawer.draw_landmarks(
                    result_img, # Source Image
                    i, # Hand Landmark To Draw
                    self.hand_Identify.HAND_CONNECTIONS, # Connect Hand Landmarks Using HandSolutionModel
                    self.drawer.DrawingSpec(color = (0, 0, 255),circle_radius=3, thickness=-1), # Draw Landmark With Red Circle
                    self.drawer.DrawingSpec(thickness=3, color=(0, 255, 0)) # Draw Connection Line
                )

        return result_img
