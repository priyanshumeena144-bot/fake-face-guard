import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time

# =========================
# Page config
# =========================
st.set_page_config(page_title="Fake Face Guard", layout="centered")

st.title("🛡 Fake Face Guard")
st.caption("Live Camera Face Detection + AI Prediction")

# =========================
# Session state
# =========================
if "run" not in st.session_state:
    st.session_state.run = False

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = ("DETECTING", 0.0)

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Models/fake_face_model.h5")

model = load_model()

# =========================
# Face detector
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# Prediction function
# =========================
def predict_face(face):
    img = cv2.resize(face, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    pred = float(model.predict(img, verbose=0)[0])

    if pred > 0.6:
        return "REAL", pred
    elif pred < 0.4:
        return "FAKE", 1 - pred
    else:
        return "UNCERTAIN", abs(pred - 0.5) * 2

# =========================
# UI Controls
# =========================
start = st.checkbox("▶ Start Camera")

FRAME = st.image([])

# =========================
# Camera logic (NO while loop)
# =========================
if start:
    cap = cv2.VideoCapture(0)  # 👈 CHANGE to 1 or 2 ONLY if needed

    if not cap.isOpened():
        st.error("❌ Camera not accessible")
    else:
        ret, frame = cap.read()

        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                st.session_state.frame_count += 1

                # 🔥 Predict only every 5 frames
                if st.session_state.frame_count % 5 == 0:
                    label, conf = predict_face(face)
                    st.session_state.last_result = (label, conf)
                else:
                    label, conf = st.session_state.last_result

                if label == "REAL":
                    color = (0, 255, 0)
                elif label == "FAKE":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

            FRAME.image(frame, channels="BGR")

        cap.release()
        time.sleep(0.03)  # FPS control
else:
    FRAME.image(np.zeros((480, 640, 3), dtype=np.uint8))
