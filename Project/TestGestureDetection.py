import cv2
import numpy as np
import DetectHand
import torch
import torch.nn as nn


class GestureDetector(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features=99, out_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=5)
        )
        
    def forward(self, x):
        x = self.head(x)
        return x

class TestGestureDetection:

    def __init__(self):
        self.action =  ['command_pose', 'volume_control', 'next', 'previous', 'play_pause', 'none']

        self.camera = cv2.VideoCapture(0)
        self.hand_detector = DetectHand.DetectHand()

        self.video_output = cv2.VideoWriter(
            filename = "Videos\\main.mp4",
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'),
            fps = int(self.camera.get(cv2.CAP_PROP_FPS)),
            frameSize = (
                int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
        )

        self.model = torch.load("Project/Models/model_best.pt")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def test_data(self):
        while True:
            return_value, frame = self.camera.read()
            
            if return_value:

                hands = self.hand_detector.detect_from_frame(frame)
                
                if hands is not None:
                    for hand in hands:

                        angle, joint = self.handle_landmark_calculation(hand)

                        # Add label
                        angle = np.array(angle, dtype=np.float32)

                        feed = np.concatenate([joint.flatten(), angle]).astype(np.float32)
                        feed = torch.from_numpy(feed).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                        prediction = self.model(feed)
                        prediction = torch.argmax(prediction).item()
                        # prediction = torch.softmax(prediction, dim=1)
                        print(self.action[prediction])
                        # print(prediction)
                        
                frame = self.hand_detector.processed_from_frame(frame)

                cv2.imshow("Detection Test", frame)
                self.video_output.write(frame)

                if cv2.waitKey(1) > 0:
                    break
            
            else:
                print("Camera Not Found")

        cv2.destroyAllWindows()
        self.camera.release()

   
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


if __name__ == "__main__":
    test = TestGestureDetection()
    test.test_data()