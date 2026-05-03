import cv2
import logging
import time

# Configure logging
logger = logging.getLogger(__name__)

class WebcamCapture:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error("Could not open webcam.")
            return False
        return True

    def get_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()

    def capture_demo(self):
        """Displays webcam feed for a few seconds as a demo."""
        if not self.start():
            return
        
        start_time = time.time()
        while time.time() - start_time < 5:  # Run for 5 seconds
            frame = self.get_frame()
            if frame is not None:
                cv2.imshow("Webcam Stress Analysis Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        self.release()
        cv2.destroyAllWindows()

def get_capture():
    return WebcamCapture()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    capture = get_capture()
    capture.capture_demo()
