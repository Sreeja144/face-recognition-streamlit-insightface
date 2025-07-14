import cv2
import numpy as np
import pickle
import os
from datetime import datetime
import smtplib
import psycopg2
from email.message import EmailMessage
from insightface.app import FaceAnalysis

# === CONFIGURATION ===
DB_URL = "postgresql://faceuser:gruqofbpAImi7EY6tyrGQjVsmMgMPiG6@dpg-d1oiqqadbo4c73b4fca0-a.frankfurt-postgres.render.com/face_db_7r21"
EMAIL_SENDER = "smadala4@gitam.in"
EMAIL_PASSWORD = "kljnnztpqqotjuwe"  
EMAIL_RECEIVER = "smadala4@gitam.in"

# === Setup ===
os.makedirs("unknown_faces", exist_ok=True)
unknown_face_paths = []

# === Load embeddings ===
with open("registered_faces.pkl", "rb") as f:
    registered_faces = pickle.load(f)
with open("blacklist_faces.pkl", "rb") as f:
    blacklist_faces = pickle.load(f)

# === Face Recognition Model ===
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# === Match function ===
def find_match(embedding, database, threshold=0.4):
    best_name = "Unknown"
    best_score = -1
    for name, db_emb in database.items():
        sim = np.dot(embedding, db_emb) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
        if sim > best_score and sim > threshold:
            best_score = sim
            best_name = name
    return best_name, best_score

# === Insert unknown face to PostgreSQL ===
def insert_into_db(image_path, score):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO unknown_faces (image_path, similarity_score) VALUES (%s, %s)",
            (image_path, score)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"[📝] Inserted into DB: {image_path}")
    except Exception as e:
        print(f"[❌] DB Insert Error: {e}")

# === Send summary email ===
def send_summary_email(paths):
    if not paths:
        print("[ℹ️] No unknown faces to report.")
        return

    msg = EmailMessage()
    msg["Subject"] = "👤 Face Recognition Summary Report"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER

    msg.set_content(
        f"""
Hi Sreeja,

🔍 Total Unknown Faces Detected: {len(paths)}

🖼️ Saved Image Paths:
{chr(10).join(paths)}

📌 You can view them in the 'unknown_faces' folder or Render DB.

Best regards,  
Your Face Recognition System
"""
    )

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("[✅] Summary email sent.")
    except Exception as e:
        print(f"[❌] Email failed: {e}")

# === Start video stream ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[❌] Camera not accessible.")
    exit()

print("[🎥] Camera started. Press 'Q' to quit and get summary email.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding
            face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            name, score = find_match(embedding, blacklist_faces)
            color = (0, 0, 255)  # red for blacklist

            if name == "Unknown":
                name, score = find_match(embedding, registered_faces)
                if name == "Unknown":
                    color = (0, 255, 255)  # yellow for unknown
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"unknown_faces/unknown_{timestamp}.jpg"
                    cv2.imwrite(filename, face_crop)
                    unknown_face_paths.append(filename)
                    insert_into_db(filename, score)
                else:
                    color = (0, 255, 0)  # green for known

            # draw label
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, str(name), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        
        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("[🛑] Quitting...")
            break

        

except KeyboardInterrupt:
    print("[⚠️] Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    send_summary_email(unknown_face_paths)
