import cv2
import numpy as np


def extract_facial_features(duration=5, show_window=False):
    """
    Lightweight facial feature extraction (Python 3.13 compatible)
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return {"eye_ratio": 0.0, "mouth_ratio": 0.0}

    eye_ratios = []
    brightness_levels = []

    frame_count = 0
    max_frames = duration * 10

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------------------------
        # Approx Eye Activity (brightness variance)
        # ---------------------------
        eye_activity = np.std(gray)
        eye_ratios.append(eye_activity)

        # ---------------------------
        # Face brightness (fatigue proxy)
        # ---------------------------
        brightness = np.mean(gray)
        brightness_levels.append(brightness)

        if show_window:
            cv2.putText(frame, f"Activity: {eye_activity:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    if len(eye_ratios) == 0:
        return {"eye_ratio": 0.0, "mouth_ratio": 0.0}

    return {
        "eye_ratio": float(np.mean(eye_ratios)),
        "mouth_ratio": float(np.mean(brightness_levels))
    }


if __name__ == "__main__":
    print(extract_facial_features())