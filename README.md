# Media Remote With Hand Gesture
Remote control device will allow the users to interact with and control their media devices using hand gestures. Gesture recognition was done through deep learning model.

## Features
- Real-time hand gesture recognition.
- Customizable data collection process.
- Customizable training process.

## Requirements
- Python 3.6 or higher
- PyTorch (`pip install torch`)
- NumPy (`pip install numpy`)
- OpenCV (`pip install opencv-python`)
- tqdm (`pip install tqdm`)

## File Explaination
- CollectData.py (Data collection from camera)
- DetectHand.py (Hand recognition and getting hand landmarks from mediapipe hand solution model)
- GestureDetection.py (Recognition of gesture and applying remote control logic)
- RunHandDetectOnly.py (Testing hand recognition)
- TestGestureDetection.py (Testing gesture recognition from custom model)
- training_model.py (Run file for creating and training model)
- configuration.py (Model and system configuation data)
- metrics.py (model accuracy logic)
- tensorboard_visualizer.py (Drawing results in tensorboard)
- trainer.py (Training and testing loops)

## Training Details
- Stochastic Gradient Descent used as optimizer
- Learing rate step milestone is used (`Iterable = (25,50,75,90)`)
- Epoch number = 100
- 70% Training Dataset
- 30% Testing Dataset

## Training Results
Currently best model found:
- Epoch number: 100
- Weight Decay: 0.0001
- Momentum: 0.95
- Learning Rate 0.002
- Learning Rate Gamma: 0.25
- Learning Rate Step Milestones: 25, 50, 75, 90
- Batch Size: 256
- Number of Workers: 1
- Seed: 42
- Model
```
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
```
