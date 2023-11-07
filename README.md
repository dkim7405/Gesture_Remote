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
```python
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

### Confusion Matrix
Beginning Model:<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/e6755489-0e86-4157-b22c-2b6d47d9ac46)

Final Model:<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/7e5b9d31-af94-4299-8a15-4edcde6ce0d9)

### Graphs
**Runs:**<br />
<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/6e97f98c-057c-4490-b4bd-cefd6b45d98d)

**Learning Rate:**<br />
<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/c7d81ca4-8d14-4206-848c-fca469af3307)

**Training Loss:**<br />
<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/afbc0f8d-ee8b-4030-9a99-9e9a54b1ee7e)

**Training Accuracy:**<br />
<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/5ee5c187-dd8d-44a0-87f0-727808fc1d22)

**Test Loss:**<br />
<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/f3f3036f-e2fd-4591-aa35-91778c1f9484)

**Test Accuracy:**<br />
<br />
![image](https://github.com/dkim7405/Gesture_Remote/assets/122648295/cb5ceb37-5def-4c0e-9d7e-e81a2772a555)

## Demo Pictures
These are demonstration of pictures for the program

### Hand Recognition
![Hand_Detection](https://github.com/dkim7405/Gesture_Remote/assets/122648295/3d96020c-5efc-4cab-af23-7b0076e15dda)

### Gesture Recognition
![Detection_Test](https://github.com/dkim7405/Gesture_Remote/assets/122648295/7f921a05-636b-4aff-822c-8c316563f100)

