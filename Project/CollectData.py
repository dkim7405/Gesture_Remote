import cv2
import mediapipe as mp
import numpy as np
import time
import os
import DetectHand

class CollectData:

    def __init__(self):
        self.action_path = ""
        self.data_amount = 100

        self.camera = cv2.VideoCapture(0)
        self.hand_detector = DetectHand.DetectHand()

        self.created_time = time.time()

    def collect_data(self, collecting_action, collecting_action_index):
        self.make_data_folder()

        while self.camera.isOpened():
            data = []
            return_value, frame = self.camera.read()
            frame = self.pre_data_collection(frame, collecting_action)
            cv2.imshow("Waiting For Collecting", frame)
            cv2.waitKey(3000)

            self.collecting_action(collecting_action, collecting_action_index, data)

            self.save_data(collecting_action, data)
            break

    def pre_data_collection(self, camera_frame, collecting_action):
        camera_frame = cv2.flip(camera_frame, 1)
        camera_frame = cv2.putText(
            img=camera_frame,
            text=f"Waiting For Collecting ... {collecting_action}",
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2
        )

        return camera_frame

    def collecting_action(self, collecting_action, collecting_action_index, data):
        start_time = time.time()

        while len(data) <= self.data_amount:
            return_value, frame = self.camera.read()

            if return_value:
                hands = self.hand_detector.detect_from_frame(frame)

                if hands is not None:
                    for hand in hands:

                        angle, joint = self.handle_landmark_calculation(hand)

                        # Add label
                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, collecting_action_index)

                        d = np.concatenate([joint.flatten(), angle_label])

                        print(d[-1])

                        data.append(d)

                frame = self.hand_detector.processed_from_frame(frame)
                cv2.imshow(f"Collecting Data ... {collecting_action} ...", frame)

                if cv2.waitKey(1) > 0:
                    break

    def handle_landmark_calculation(self, hand):
        # 21 Landmarks, 3 Coordinates (x, y, z)
        joint = np.zeros((21, 4))

        for joint_index, joint_landmark in enumerate(hand.landmark):
            joint[joint_index] = [joint_landmark.x, joint_landmark.y, joint_landmark.z, joint_landmark.visibility]

        # Compute angles between joints
        temp_vector1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent Joint
        temp_vector2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child Joint
        temp_vector3 = temp_vector2 - temp_vector1  # Vector between joints

        # Normalize the vector
        temp_vector3 = temp_vector3 / np.linalg.norm(temp_vector3, axis=1)[:, np.newaxis]  # Directional Vector

        # Compute angle using arccos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                                   temp_vector3[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                   temp_vector3[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

        # Convert to degree
        angle = np.degrees(angle)

        return angle, joint

    def save_data(self, action, data):

        # Define the file path
        raw_file_path = os.path.join(self.action_path, f"{action}.npy")

        if os.path.exists(raw_file_path):
            existing_data = np.load(raw_file_path)
            data = np.vstack((existing_data, np.array(data)))

        # Save the updated data to the file
        np.save(raw_file_path, data)

    def make_data_folder(self):
        self.action_path = f"Project/TrainingModel/data"
        os.makedirs(self.action_path, exist_ok=True)


if __name__ == "__main__":
    actions = ['command_pose', 'volume_control', 'next', 'previous', 'play_pause', 'none']

    collect_data = CollectData()
    collect_data.collect_data(actions[1], 1)