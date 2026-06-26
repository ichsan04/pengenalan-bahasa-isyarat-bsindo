from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import Counter

app = Flask(__name__)

# =========================
# MODELS
# =========================
MODELS = {
    "huruf": {
        "model": joblib.load("model_huruf1.pkl"),
        "scaler": joblib.load("scaler_huruf1.pkl"),
        "encoder": joblib.load("encoder_huruf1.pkl"),
        "threshold": 0.7,
        "use_scaler": True
    },
    "angka": {
        "model": joblib.load("model_angka.pkl"),
        "scaler": joblib.load("scaler_angka.pkl"),
        "encoder": joblib.load("encoder_angka.pkl"),
        "threshold": 0.7,
        "use_scaler": True
    },
    "kata": {
        "model": joblib.load("model_kata.pkl"),
        "encoder": joblib.load("encoder_kata.pkl"),
        "threshold": 0.5,
        "use_scaler": False
    }
}

current_mode = "huruf"

latest_prediction = "-"
latest_confidence = 0
latest_hands = 0
fps = 0

pred_buffer = []

# =========================
# CAMERA FIX
# =========================
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(3, 640)
camera.set(4, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# =========================
# MEDIAPIPE
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# =========================
# STREAM
# =========================
def generate_frames():

    global latest_prediction, latest_confidence, latest_hands, fps, pred_buffer, current_mode

    prev_time = time.time()

    while True:

        success, frame = camera.read()
        if not success or frame is None:
            continue

        frame = cv2.flip(frame, 1)

        # FPS
        now = time.time()
        fps = int(1 / max(now - prev_time, 0.001))
        prev_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        latest_hands = 0

        if result.multi_hand_landmarks:

            latest_hands = len(result.multi_hand_landmarks)

            for h in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

            # =========================
            # FEATURE BUILDER (FIX 63 / 126)
            # =========================
            hand_count = len(result.multi_hand_landmarks)
            landmarks = []

            # HAND 1
            if hand_count >= 1:
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0] * 63)

            # HAND 2
            if hand_count >= 2:
                for lm in result.multi_hand_landmarks[1].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            # =========================
            # AUTO SHAPE FIX
            # =========================
            if current_mode in ["huruf", "angka"]:
                X = np.array(landmarks[:63], dtype=np.float32).reshape(1, 63)
            else:
                # kata
                if len(landmarks) < 126:
                    landmarks.extend([0.0] * (126 - len(landmarks)))

                X = np.array(landmarks[:126], dtype=np.float32).reshape(1, 126)

            cfg = MODELS[current_mode]

            # scaler only huruf & angka
            if cfg["use_scaler"]:
                X = cfg["scaler"].transform(X)

            probs = cfg["model"].predict_proba(X)[0]
            idx = np.argmax(probs)

            prediction = cfg["encoder"].inverse_transform([idx])[0]
            confidence = float(probs[idx])

            if confidence < cfg["threshold"]:
                prediction = "Tidak Dikenali"

            pred_buffer.append(prediction)
            if len(pred_buffer) > 10:
                pred_buffer.pop(0)

            latest_prediction = Counter(pred_buffer).most_common(1)[0][0]
            latest_confidence = confidence

        else:
            latest_prediction = "-"
            latest_confidence = 0
            latest_hands = 0
            pred_buffer.clear()

        # =========================
        # UI
        # =========================
        cv2.putText(frame, f"MODE: {current_mode}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame, f"PRED: {latest_prediction}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"CONF: {latest_confidence*100:.1f}%", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, f"HAND: {latest_hands} FPS: {fps}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # encode frame
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/set_mode/<mode>")
def set_mode(mode):
    global current_mode, pred_buffer

    if mode in MODELS:
        current_mode = mode
        pred_buffer.clear()

    return jsonify({"mode": current_mode})


@app.route("/stats")
def stats():
    return jsonify({
        "prediction": latest_prediction,
        "confidence": round(latest_confidence * 100, 1),
        "hands": latest_hands,
        "fps": fps,
        "mode": current_mode
    })


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True
    )