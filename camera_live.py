import cv2
import time
import numpy as np
import tensorflow as tf
from collections import deque

# ==============================
# Load face detector
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# Load trained model
# ==============================
model = tf.keras.models.load_model("Models/fake_face_model.h5")

# ==============================
# Open Camera (OBS Virtual Camera / Webcam)
# ==============================
cap = cv2.VideoCapture(0)
time.sleep(2)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera opened successfully")

# ==============================
# Prediction smoothing buffer (STEP 5.5)
# ==============================
PRED_BUFFER = deque(maxlen=5)

# ==============================
# Prediction function (STEP 3 + 5.3)
# ==============================
def predict_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = img.reshape(1, 224, 224, 3)

    pred = float(model.predict(img, verbose=0)[0])
    PRED_BUFFER.append(pred)

    avg_pred = sum(PRED_BUFFER) / len(PRED_BUFFER)

    if avg_pred > 0.7:
        return "REAL", avg_pred
    elif avg_pred < 0.3:
        return "FAKE", 1 - avg_pred
    else:
        return "UNCERTAIN", abs(avg_pred - 0.5) * 2

# ==============================
# Live loop
# ==============================
frame_count = 0
last_label = "DETECTING"
last_conf = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        frame_count += 1

        # 🔥 Predict only every 5 frames (FPS improve – STEP 5.3)
        if frame_count % 5 == 0:
            label, conf = predict_frame(face)
            last_label = label
            last_conf = conf
        else:
            label, conf = last_label, last_conf

        # ==============================
        # Threshold + color logic
        # ==============================
        if conf < 0.6:
            label = "UNCERTAIN"
            color = (0, 255, 255)   # yellow
        elif label == "REAL":
            color = (0, 255, 0)     # green
        else:
            color = (0, 0, 255)     # red

        # ==============================
        # Draw box + label
        # ==============================
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

    cv2.imshow("Fake Face Guard - LIVE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
