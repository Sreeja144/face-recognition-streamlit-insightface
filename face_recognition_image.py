import cv2
import numpy as np
import pickle
from datetime import datetime
from insightface.app import FaceAnalysis
import streamlit as st

# === CONFIG ===
image_path = "faces/political3.png"

# === Load Embeddings ===
with open("registered_faces.pkl", "rb") as f:
    registered_faces = pickle.load(f)

with open("blacklist_faces.pkl", "rb") as f:
    blacklist_faces = pickle.load(f)

# === Load and Analyze Image ===
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 480))

frame = cv2.imread(image_path)
if frame is None:
    print("[❌] Image not found!")
    exit()

faces = app.get(frame)

# === Find Best Match ===
def find_match(embedding, face_dict, threshold=0.4):
    best_match = "Unknown"
    best_score = -1

    for name, emb in face_dict.items():
        sim = np.dot(embedding, emb) / (
            np.linalg.norm(embedding) * np.linalg.norm(emb)
        )
        if sim > best_score and sim > threshold:
            best_score = sim
            best_match = name
    return best_match

# === Draw and Log ===
for face in faces:
    bbox = face.bbox.astype(int)
    embedding = face.embedding
    face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    name = find_match(embedding, blacklist_faces)
    if name != "Unknown":
        color = (0, 0, 255)  # Red
        label = f"[BLACKLIST] {name}"
    else:
        name = find_match(embedding, registered_faces)
        if name != "Unknown":
            color = (0, 255, 0)  # Green
            label = f"[KNOWN] {name}"
        else:
            color = (0, 255, 255)  # Yellow
            label = "[UNKNOWN]"

    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# === Show Result ===
cv2.imshow("Screenshot Recognition", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
