# src/face_iris_capture.py
import cv2
import mediapipe as mp
import time
import csv
import numpy as np
from pathlib import Path

# Output CSV path
OUT_CSV = Path("data/eye_landmarks.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# MediaPipe setup (refine_landmarks=True gives iris landmarks)
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True,  # important for iris
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Iris landmark indices used by MediaPipe:
LEFT_IRIS = [468, 469, 470, 471, 472]   # left eye iris landmarks
RIGHT_IRIS = [473, 474, 475, 476, 477]  # right eye iris landmarks

# Helpful function: compute average normalized (x,y) of landmark indices
def avg_landmark_coords(landmarks, indices):
    pts = [(landmarks[i].x, landmarks[i].y) for i in indices]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(np.mean(xs)), float(np.mean(ys)))

# CSV header
if not OUT_CSV.exists():
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "left_x", "left_y", "right_x", "right_y"])

cap = cv2.VideoCapture(0)
prev_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for a mirror-like view (optional)
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        left_center = right_center = (None, None)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Compute iris centers (normalized coordinates 0..1)
            left_center = avg_landmark_coords(face_landmarks, LEFT_IRIS)
            right_center = avg_landmark_coords(face_landmarks, RIGHT_IRIS)

            # Convert normalized to pixel for drawing
            lx, ly = int(left_center[0] * w), int(left_center[1] * h)
            rx, ry = int(right_center[0] * w), int(right_center[1] * h)

            # Draw circle on iris centers
            cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)
            cv2.circle(frame, (rx, ry), 3, (0, 255, 0), -1)

            # Draw face mesh (optional, slows a bit)
            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0],
                                      mp_face.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawing_spec)

            # Save normalized coords to CSV (timestamp in seconds)
            ts = time.time()
            with open(OUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ts, left_center[0], left_center[1],
                                 right_center[0], right_center[1]])

            # Show normalized values on frame (for debug)
            cv2.putText(frame, f"L: {left_center[0]:.3f},{left_center[1]:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"R: {right_center[0]:.3f},{right_center[1]:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Show FPS
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time else 0.0
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.imshow("FaceMesh Iris Capture (press q to quit)", frame)

        # exit conditions
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        # also break if window closed by user
        if cv2.getWindowProperty("FaceMesh Iris Capture (press q to quit)", cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Saved to:", OUT_CSV.absolute())