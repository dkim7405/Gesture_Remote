import cv2
import DetectHand

class RunHandDetectOnly:
    
    def __init__(self):
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

    def run_hand_detection(self):
        while True:
            return_value, frame = self.camera.read()
            
            if return_value:
                frame = self.hand_detector.processed_from_frame(frame)

                cv2.imshow("Hand Detection", frame)
                self.video_output.write(frame)

                if cv2.waitKey(1) > 0:
                    break
            
            else:
                print("Camera Not Found")

        cv2.destroyAllWindows()
        self.camera.release()


if __name__ == "__main__":
    program = RunHandDetectOnly()
    program.run_hand_detection()