import cv2
import numpy as np


# ---------------------------
# Helper Functions
# ---------------------------
def compute_activity(gray_frame):
    """
    Measures intensity variation (proxy for eye movement / alertness)
    """
    return np.std(gray_frame)


def compute_brightness(gray_frame):
    """
    Measures average brightness (proxy for fatigue / dullness)
    """
    return np.mean(gray_frame)


# ---------------------------
# Main Extraction Function
# ---------------------------
def extract_facial_features(duration=5, show_window=False):
    """
    Lightweight facial feature extraction (Python 3.13 compatible)

    Returns:
        {
            "eye_ratio": float,
            "mouth_ratio": float
        }
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Webcam not accessible")
        return {
            "eye_ratio": 0.0,
            "mouth_ratio": 0.0
        }

    activity_values = []
    brightness_values = []

    frame_count = 0
    max_frames = duration * 10  # approx FPS scaling

    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---------------------------
            # Feature Computation
            # ---------------------------
            activity = compute_activity(gray)
            brightness = compute_brightness(gray)

            activity_values.append(activity)
            brightness_values.append(brightness)

            # ---------------------------
            # Optional Display
            # ---------------------------
            if show_window:
                cv2.putText(frame, f"Activity: {activity:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.putText(frame, f"Brightness: {brightness:.2f}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Facial Feature Capture", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

    except Exception as e:
        print("⚠️ Facial capture error:", str(e))

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    # ---------------------------
    # Aggregate Results
    # ---------------------------
    if len(activity_values) == 0:
        return {
            "eye_ratio": 0.0,
            "mouth_ratio": 0.0
        }

    return {
        "eye_ratio": float(np.mean(activity_values)),
        "mouth_ratio": float(np.mean(brightness_values))
    }


# ---------------------------
# Test Run
# ---------------------------
if __name__ == "__main__":
    features = extract_facial_features(duration=5, show_window=True)

    print("\nFacial Features Extracted:\n")
    print(features)