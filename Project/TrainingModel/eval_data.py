import torch
import os
import cv2
import mediapipe as mp
import numpy as np
import torch.nn as nn

from trainer import configuration

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

class EvalData:
    def __init__(self):

        self.drawer = mp.solutions.drawing_utils
        self.hand_Identify = mp.solutions.hands

        self.data_loaded = np.concatenate([
            np.load(configuration.DatasetConfig.command_pose_path),
            np.load(configuration.DatasetConfig.volume_control_path),
            np.load(configuration.DatasetConfig.next_path),
            np.load(configuration.DatasetConfig.previous_path),
            np.load(configuration.DatasetConfig.play_pause_path),
            np.load(configuration.DatasetConfig.none_path)
        ])

        self.data_points = self.data_loaded[:, :-1].astype(np.float32)
        self.data_labels = self.data_loaded[:, -1].astype(np.int64)

        self.model = torch.load("Project/Models/model_best.pt")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def eval_data(self):

        for i in range(len(self.data_points)):
            data_point = self.data_points[i]
            data_point = torch.from_numpy(data_point).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

            prediction = self.model(data_point)
            prediction = torch.argmax(prediction).item()

            if(prediction != self.data_labels[i]):
                os.makedirs(f"Project/TrainingModel/eval_data", exist_ok=True)
                # create me an empty image
                img = np.zeros((224,224,3), np.uint8)
                data_point = data_point.cpu().numpy()
                data_point = data_point.reshape(99)
                data_point_without_angle = data_point[:84]
                joint = data_point_without_angle.reshape(21,4)

                temp_vector1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent Joint
                temp_vector2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child Joint

                # draw a hand using the information is taken from mp.solutions.hands but with joint variable
                for i in range(21):
                    x = int(joint[i][0] * 224)
                    y = int(joint[i][1] * 224)
                    cv2.circle(img, (x,y), 3, (0,0,255), -1)

                # connect the joints to make a hand using temp_vector1 and temp_vector2 where joint variable is the joint location and temp_vector1 and temp_vector2 are the indecies of joints to connect
                for i in range(20):
                    x1 = int(temp_vector1[i][0] * 224)
                    y1 = int(temp_vector1[i][1] * 224)
                    x2 = int(temp_vector2[i][0] * 224)
                    y2 = int(temp_vector2[i][1] * 224)
                    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
                
                # save the image
                cv2.imwrite(f"Project/TrainingModel/eval_data/{i}.jpg", img)


if __name__ == "__main__":
    eval_data = EvalData()
    eval_data.eval_data()