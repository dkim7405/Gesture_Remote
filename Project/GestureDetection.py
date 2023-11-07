import cv2
import numpy as np
import DetectHand
import torch
import torch.nn as nn

class GestureDetector(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_features=99, out_features=99),
            nn.Linear(in_features=99, out_features=99),
            nn.Linear(in_features=99, out_features=99),
            nn.Linear(in_features=99, out_features=99),
            nn.Linear(in_features=99, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=16),
            nn.Linear(in_features=16, out_features=6)
        )
        
    def forward(self, x):
        x = self.head(x)
        return x

class GestureDetection:

    def __init__(self):
        self.actions = ['command_pose', 'volume_control', 'next', 'previous', 'play_pause', 'none']
        self.action_check_sequence = []
        self.user_seq = []
        self.action = 'none'
        self.command = 'none'
        self.command_on = False
        self.volume_cmd_on = False

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

    def detect_from_cam(self):
        while True:
            return_value, frame = self.camera.read()
            
            if return_value:

                hands = self.hand_detector.detect_from_frame_normalized(frame)
                
                if hands is not None:
                    for hand in hands:

                        angle, joint = self.handle_landmark_calculation(hand)

                        # Add label
                        angle = np.array(angle, dtype=np.float32)

                        feed = np.concatenate([joint.flatten(), angle]).astype(np.float32)
                        feed = torch.from_numpy(feed).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

                        prediction = self.model(feed)
                        self.action = self.get_action(prediction)

                        if self.action != 'none':
                            self.user_seq.append(self.action)

                self.remote_control_check()

                frame = self.hand_detector.processed_from_frame(frame)

                cv2.putText(frame, self.command, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

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
    
    def get_action(self, model_prediction):
        action = 'none'
        prediction_index = torch.argmax(model_prediction).item()
        prediction_confidence_list = torch.softmax(model_prediction, dim=1).squeeze(0).cpu().detach().numpy()
        prediction_confidence = prediction_confidence_list[prediction_index]

        if prediction_confidence > 0.8:
            self.action_check_sequence.append(prediction_index)

            if len(self.action_check_sequence) == 3:
                if (self.action_check_sequence[0] == self.action_check_sequence[1] == self.action_check_sequence[2])\
                and (self.actions[self.action_check_sequence[0]] != 'none'):
                    action = self.actions[self.action_check_sequence[0]]
                else:
                    action = 'none'
                
                self.action_check_sequence = self.action_check_sequence[1:]

        return action
    
    def remote_control_check(self):

        print(self.user_seq)

        if len(self.user_seq) == 2:

            if (self.user_seq[0] == 'volume_control' and self.user_seq[1] == 'play_pause' and self.command == 'volume_up')\
                or (self.user_seq[0] == 'command_pose' and self.user_seq[1] == 'play_pause' and self.command == 'volume_down'):                 
                self.command = 'volume_stop'
        
            elif self.user_seq[0] == 'volume_control':
                if self.user_seq[1] == 'command_pose' and self.command != 'volume_up':
                    self.command = 'volume_down'
                    self.volume_cmd_on = True

            elif self.user_seq[0] == 'command_pose':
                self.command = 'command'
                self.command_on = True
                self.volume_cmd_on = False
                if self.user_seq[1] == 'volume_control':
                    self.command = 'volume_up'
                    self.volume_cmd_on = True
                elif self.user_seq[1] == 'play_pause':
                    self.command = 'play_pause'
                elif self.user_seq[1] == 'next':
                    self.command = 'next'
                elif self.user_seq[1] == 'previous':
                    self.command = 'previous'
            
            elif self.command_on and not self.volume_cmd_on and self.user_seq[0] == self.user_seq[1]:
                self.command = self.user_seq[1]
                self.command_on = False
                self.volume_cmd_on = False

            print(self.command)
            self.user_seq = self.user_seq[1:]
            
 
if __name__ == "__main__":
    gesture_detection = GestureDetection()
    gesture_detection.detect_from_cam()