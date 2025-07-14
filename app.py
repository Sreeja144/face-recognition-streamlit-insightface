import os
os.environ['STREAMLIT_SERVER_PORT'] = '8000'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

import streamlit as st
st.set_page_config(page_title="Face Detection", layout="wide")  # ✅ MUST BE FIRST

import cv2
import numpy as np
import pickle
import smtplib
import psycopg2
import time
from datetime import datetime
from email.message import EmailMessage
from insightface.app import FaceAnalysis

# === CONFIGURATION ===
VIDEO_PATH = "video/classroom.mp4"
EMAIL_SENDER = "smadala4@gitam.in"
EMAIL_PASSWORD = "kljn nztp qqot juwe"
DB_URL = "postgresql://faceuser:gruqofbpAImi7EY6tyrGQjVsmMgMPiG6@dpg-d1oiqqadbo4c73b4fca0-a.frankfurt-postgres.render.com/face_db_7r21"

# === ✅ DEBUG VIDEO DIRECTORY ===
st.write("✅ Current working directory:", os.getcwd())
if os.path.exists("video"):
    st.write("📁 'video' folder contents:", os.listdir("video"))
else:
    st.write("❌ 'video' folder not found. Please push it to Git.")

# === Setup folders ===
os.makedirs("unknown_faces", exist_ok=True)

# === Load Embeddings ===
with open("registered_faces.pkl", "rb") as f:
    registered_faces = pickle.load(f)
with open("blacklist_faces.pkl", "rb") as f:
    blacklist_faces = pickle.load(f)

# === Streamlit UI ===
st.title("🎥 Face Detection System with Email & DB")

frame_display = st.empty()
status_msg = st.empty()
email_icon = st.empty()

start_btn = st.button("▶️ Start Detection")

if "email_sent" not in st.session_state:
    st.session_state.email_sent = False
if "final_email" not in st.session_state:
    st.session_state.final_email = ""
if "unknown_faces" not in st.session_state:
    st.session_state.unknown_faces = []

# === Helpers ===
seen_unknown_embeddings = []

def is_duplicate(embedding, seen_list, threshold=0.6):
    for emb in seen_list:
        sim = np.dot(embedding, emb) / (np.linalg.norm(embedding) * np.linalg.norm(emb))
        if sim > threshold:
            return True
    return False

def find_match(embedding, face_db, threshold=0.45):
    best_name = None
    best_score = -1
    for name, db_emb in face_db.items():
        sim = np.dot(embedding, db_emb) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
        if sim > best_score and sim > threshold:
            best_name, best_score = name, sim
    return best_name, best_score

def insert_to_db(path, score):
    try:
        with psycopg2.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO unknown_faces (image_path, similarity_score) VALUES (%s, %s)", (path, score))
                conn.commit()
    except Exception as e:
        st.error(f"❌ DB Error: {e}")

def send_email_with_images(image_paths, to_email):
    if not to_email or not image_paths:
        return
    msg = EmailMessage()
    msg["Subject"] = "🔍 Face Detection Report"
    msg["From"] = EMAIL_SENDER
    msg["To"] = to_email
    msg.set_content(f"""
Hi,

Some unknown faces were detected in the classroom video.

- Attached are a few images.
- Stored in 'unknown_faces' folder and logged to DB.

Thanks,
Face Detection System
""")
    for path in image_paths[:5]:
        with open(path, 'rb') as img:
            msg.add_attachment(img.read(), maintype='image', subtype='jpeg', filename=os.path.basename(path))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        st.toast("✅ Email sent successfully!", icon="📨")
        st.success(f"📧 Sent to {to_email}")
    except Exception as e:
        st.error("❌ Email Failed")
        st.code(str(e))

# === DETECTION ===
if start_btn:
    app = FaceAnalysis(name="buffalo_s", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(VIDEO_PATH)
    st.write("🎥 Video opened?", cap.isOpened())

    if not cap.isOpened():
        st.error("❌ Could not open video.")
        st.stop()

    unknown_faces = []
    frame_skip = 70

    while True:
        for _ in range(frame_skip):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding

            name, score = find_match(emb, registered_faces)
            color = (0, 255, 0) if name else (0, 0, 255)

            if not name:
                name, score = find_match(emb, blacklist_faces)
                if not name:
                    if not is_duplicate(emb, seen_unknown_embeddings):
                        name = "Unknown"
                        color = (0, 255, 255)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        path = f"unknown_faces/unknown_{ts}.jpg"
                        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        cv2.imwrite(path, crop)
                        insert_to_db(path, -1.0)
                        unknown_faces.append(path)
                        seen_unknown_embeddings.append(emb)
                    else:
                        name = "Unknown"
                        color = (0, 255, 255)

            label = name
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_display.image(rgb_frame, channels="RGB")
        time.sleep(0.0005)

    cap.release()
    st.session_state.unknown_faces = unknown_faces

# === Email Form ===
with st.form("final_email_form"):
    st.markdown("#### 📩 Enter your email to receive summary report")
    st.session_state.final_email = st.text_input("Email")
    submit = st.form_submit_button("Send Report")
    if submit and not st.session_state.email_sent:
        send_email_with_images(st.session_state.unknown_faces, st.session_state.final_email)
        st.session_state.email_sent = True

# === Floating Icon ===
with email_icon.container():
    st.markdown("<div style='position: fixed; bottom: 20px; right: 30px; font-size: 24px;'>📨</div>", unsafe_allow_html=True)
