import cv2
import numpy as np
import pickle
import threading
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# Load registered face embeddings (dictionary: {name: embedding})
with open("registered_faces.pkl", "rb") as f:
    known_faces = pickle.load(f)  # {name: embedding}

# Ensure output folders exist
os.makedirs("blacklist_images", exist_ok=True)

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Save attendance
def mark_attendance(name, cam_id):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"attendance_cam{cam_id}.csv", "a") as f:
        f.write(f"{name},{now},cam{cam_id}\n")

# Save blacklist log and cropped face
def mark_blacklist(name, face_crop, cam_id):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"blacklist_cam{cam_id}.csv", "a") as f:
        f.write(f"{name},{timestamp},cam{cam_id}\n")
    img_path = f"blacklist_images/{name}_cam{cam_id}_{timestamp}.jpg"
    cv2.imwrite(img_path, face_crop)

# Match face embedding
def find_match(face_embedding, known_faces, threshold=0.5):
    best_match = None
    best_sim = -1
    for name, db_embedding in known_faces.items():  # ✅ db_embedding is direct np.array
        sim = np.dot(face_embedding, db_embedding) / (
            np.linalg.norm(face_embedding) * np.linalg.norm(db_embedding)
        )
        if sim > best_sim and sim > threshold:
            best_sim = sim
            best_match = name
    return best_match if best_match else "Unknown"

# Process camera/video
def process_camera(video_path, cam_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[❌] Error opening {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            name = find_match(embedding, known_faces)
            face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            if name == "Unknown":
                color = (0, 255, 255)  # Yellow
            elif name.lower() in ["kiranmai", "lahari", "tejakshara", "pavan"]:
                color = (0, 0, 255)  # Red
                mark_blacklist(name, face_crop, cam_id)
            else:
                color = (0, 255, 0)  # Green
                mark_attendance(name, cam_id)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow(f"Camera {cam_id}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(f"Camera {cam_id}")

# Paths to videos
video_paths = ["videos/classroom.mp4", "videos/classroom2.mp4"]

# Start threads
threads = []
for idx, path in enumerate(video_paths):
    t = threading.Thread(target=process_camera, args=(path, idx))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
