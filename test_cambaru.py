import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import Counter

# ==========================
# PILIH MODE
# ==========================

print("\n====================")
print("PILIH MODE")
print("====================")
print("1. Huruf (A-Z)")
print("2. Angka (0-9)")
print("3. 5 Kata BISINDO")
print("====================")

pilihan = input("\nMasukkan pilihan (1/2/3): ")

# ==========================
# LOAD MODEL
# ==========================

if pilihan == "1":

    model = joblib.load("model_huruf1.pkl")
    scaler = joblib.load("scaler_huruf1.pkl")
    encoder = joblib.load("encoder_huruf1.pkl")

    judul = "Pengenalan Huruf"
    use_scaler = True

elif pilihan == "2":

    model = joblib.load("model_angka.pkl")
    scaler = joblib.load("scaler_angka.pkl")
    encoder = joblib.load("encoder_angka.pkl")

    judul = "Pengenalan Angka"
    use_scaler = True

elif pilihan == "3":

    model = joblib.load("model_5kata.pkl")
    encoder = joblib.load("encoder_5kata.pkl")

    judul = "Pengenalan 5 Kata BISINDO"
    use_scaler = False

else:
    print("Pilihan tidak valid!")
    exit()

# ==========================
# MEDIAPIPE
# ==========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pred_buffer = []
stable_prediction = "-"
confidence = 0.0

# ==========================
# DETEKSI TANGAN
# ==========================

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        confidence = 0.0

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            landmarks = []

            for lm in hand.landmark:
                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            X = np.array(landmarks).reshape(1, -1)

            if use_scaler:
                X = scaler.transform(X)

            probs = model.predict_proba(X)[0]

            idx = np.argmax(probs)

            prediction = encoder.inverse_transform([idx])[0]

            confidence = probs[idx]

            # ==========================
            # THRESHOLD
            # ==========================

            if pilihan == "3":
                threshold = 0.50
            else:
                threshold = 0.70

            if confidence < threshold:
                prediction = "Tidak Dikenali"

            # ==========================
            # STABILKAN PREDIKSI
            # ==========================

            pred_buffer.append(prediction)

            if len(pred_buffer) > 10:
                pred_buffer.pop(0)

            stable_prediction = Counter(
                pred_buffer
            ).most_common(1)[0][0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        else:

            pred_buffer.clear()
            stable_prediction = "-"
            confidence = 0.0

        # ==========================
        # DISPLAY
        # ==========================

        cv2.putText(
            frame,
            judul,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Prediksi : {stable_prediction}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Confidence : {confidence:.2f}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        if pilihan == "3":

            cv2.putText(
                frame,
                "Bingung | Kemana | Melihat",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Menulis | Okey",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "ESC = Keluar",
            (20, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(judul, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()