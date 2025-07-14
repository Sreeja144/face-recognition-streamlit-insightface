import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from insightface.app import FaceAnalysis

# Load known faces
with open("registered_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)

# Define blacklist names (lowercase for comparison)
blacklist = ["kiranmai", "lahari", "tejakshara", "pavan"]

# Create folders if not exist
os.makedirs("unknown_faces", exist_ok=True)
os.makedirs("blacklist_captures", exist_ok=True)


# Initialize face analysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Attendance file
attendance_file = "attendance.csv"

# Save attendance
def mark_attendance(name):
    with open(attendance_file, "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{name},{now}\n")

# Save unknown face
def save_unknown(face_crop):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"unknown_faces/unknown_{now}.jpg", face_crop)

# Save blacklist face
def mark_blacklist(name, face_crop):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open("blacklist_log.csv", "a") as f:
        f.write(f"{name},{now}\n")
    cv2.imwrite(f"blacklist_captures/{name}_{now}.jpg", face_crop)

# Find best match
def find_match(embedding, known_faces, threshold=0.5):
    best_match = None
    best_sim = -1
    for name, db_embedding in known_faces.items():
        sim = np.dot(embedding, db_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
        )
        if sim > best_sim and sim > threshold:
            best_sim = sim
            best_match = name
    return best_match if best_match else "Unknown"

# ==== MAIN VIDEO HANDLING ====
video_path = "videos/classroom3.1.mp4"  # Update this to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[❌] Error opening video file.")
    exit()

frame_skip = 1  # Change to 2 or 3 to skip more frames for smoother video
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    faces = app.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        name = find_match(embedding, known_faces)
        face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if name == "Unknown":
            color = (0, 255, 255)  # Yellow
            save_unknown(face_crop)
        elif name.lower() in blacklist:
            color = (0, 0, 255)  # Red
            mark_blacklist(name, face_crop)
        else:
            color = (0, 255, 0)  # Green
            mark_attendance(name)

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition Video", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
