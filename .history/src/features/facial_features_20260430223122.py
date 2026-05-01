import cv2
import numpy as np

# ---------------------------
# Safe MediaPipe Import (CRITICAL FIX)
# ---------------------------
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except Exception:
    try:
        from mediapipe.python.solutions import face_mesh as mp_face_mesh
    except Exception:
        raise ImportError("MediaPipe not installed correctly or incompatible Python version")


# ---------------------------
# Helper Functions
# ---------------------------
def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]

    hor = euclidean_dist(p1, p4)
    ver = euclidean_dist(p2, p6) + euclidean_dist(p3, p5)

    return ver / (2.0 * hor + 1e-6)


def mouth_aspect_ratio(landmarks, mouth_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in mouth_indices]

    hor = euclidean_dist(p1, p4)
    ver = euclidean_dist(p2, p6) + euclidean_dist(p3, p5)

    return ver / (2.0 * hor + 1e-6)


# ---------------------------
# Main Extraction Function
# ---------------------------
def extract_facial_features(duration=5, show_window=True):
    """
    Captures webcam data and returns averaged facial features
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Webcam not accessible")
        return {"eye_ratio": 0.0, "mouth_ratio": 0.0}

    # Landmark indices
    eye_left = [33, 160, 158, 133, 153, 144]
    eye_right = [362, 385, 387, 263, 373, 380]
    mouth = [61, 81, 13, 291, 311, 308]

    ear_values = []
    mar_values = []

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        ) as face_mesh:

            frame_count = 0
            max_frames = duration * 10  # approx FPS

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]

                    h, w, _ = frame.shape

                    landmarks = [
                        (int(lm.x * w), int(lm.y * h))
                        for lm in face_landmarks.landmark
                    ]

                    # Compute EAR
                    ear_left_val = eye_aspect_ratio(landmarks, eye_left)
                    ear_right_val = eye_aspect_ratio(landmarks, eye_right)
                    ear = (ear_left_val + ear_right_val) / 2.0

                    # Compute MAR
                    mar = mouth_aspect_ratio(landmarks, mouth)

                    ear_values.append(ear)
                    mar_values.append(mar)

                    if show_window:
                        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if show_window:
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
    if len(ear_values) == 0:
        return {"eye_ratio": 0.0, "mouth_ratio": 0.0}

    return {
        "eye_ratio": float(np.mean(ear_values)),
        "mouth_ratio": float(np.mean(mar_values))
    }


# ---------------------------
# Test Run
# ---------------------------
if __name__ == "__main__":
    features = extract_facial_features(duration=5)

    print("\nFacial Features Extracted:\n")
    print(features)