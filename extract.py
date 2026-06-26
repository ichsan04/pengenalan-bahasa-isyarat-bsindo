import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

DATASET_PATH = "dataset/dataset kata"
OUTPUT_CSV = "datasetkata1.csv"

mp_hands = mp.solutions.hands

data = []

total_images = 0
detected = 0
failed = 0

os.makedirs("debug_failed", exist_ok=True)

# =========================
# AUGMENTASI LEBIH STABIL
# =========================
def augment(row):
    row = np.array(row, dtype=np.float32)

    noise = np.random.normal(0, 0.002, row.shape)  # kecilkan noise
    scale = np.random.uniform(0.98, 1.02)          # jangan terlalu ekstrem

    return (row * scale + noise).tolist()


with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,              # ✔ FIX: 2 tangan
    model_complexity=0,           # ✔ lebih ringan & stabil
    min_detection_confidence=0.5, # ✔ FIX: jangan terlalu rendah
    min_tracking_confidence=0.5
) as hands:

    for label in os.listdir(DATASET_PATH):

        class_dir = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(class_dir):
            continue

        label = label.strip().lower()  # ✔ FIX penting

        print(f"\nProcessing {label}...")

        for img_name in os.listdir(class_dir):

            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total_images += 1

            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                failed += 1
                continue

            img = cv2.resize(img, (640, 640))
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:

                detected += 1

                row = []

                # =========================
                # AMBIL SEMUA HAND
                # =========================
                for hand_landmarks in result.multi_hand_landmarks:

                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                # =========================
                # NORMALISASI PANJANG (2 HAND = 126)
                # =========================
                while len(row) < 126:
                    row.extend([0.0, 0.0, 0.0])

                row = row[:126]

                # =========================
                # DATA ASLI
                # =========================
                data.append(row + [label])

                # =========================
                # AUGMENTASI (lebih aman)
                # =========================
                data.append(augment(row) + [label])
                data.append(augment(row) + [label])

            else:
                failed += 1
                cv2.imwrite(f"debug_failed/{label}_{img_name}", img)


# =========================
# SAVE CSV
# =========================
columns = []
for i in range(42):  # 2 tangan x 21 landmark
    columns.extend([f"x{i}", f"y{i}", f"z{i}"])

columns.append("label")

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("\n======================")
print("Total gambar :", total_images)
print("Terdeteksi   :", detected)
print("Gagal        :", failed)
print("CSV samples  :", len(df))
print("======================")