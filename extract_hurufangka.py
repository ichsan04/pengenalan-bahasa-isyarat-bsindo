import os
import cv2
import mediapipe as mp
import pandas as pd

# ==========================
# PATH
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset/dataset angka")
OUTPUT_CSV = os.path.join(BASE_DIR, "datasetangka.csv")

print("Dataset path :", DATASET_PATH)
print("Path exists  :", os.path.exists(DATASET_PATH))

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan:\n{DATASET_PATH}"
    )

# ==========================
# MEDIAPIPE
# ==========================
mp_hands = mp.solutions.hands

data = []

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
) as hands:

    for label in os.listdir(DATASET_PATH):

        class_dir = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(class_dir):
            continue

        print(f"\nProcessing class: {label}")

        success = 0
        failed = 0

        for img_name in os.listdir(class_dir):

            img_path = os.path.join(class_dir, img_name)

            img = cv2.imread(img_path)

            if img is None:
                failed += 1
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:

                hand = result.multi_hand_landmarks[0]

                row = []

                # 21 landmark × (x,y,z)
                for lm in hand.landmark:
                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                row.append(label)

                data.append(row)

                success += 1

            else:
                failed += 1

        print(f"Detected : {success}")
        print(f"Failed   : {failed}")

# ==========================
# DATAFRAME
# ==========================
columns = []

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

columns.append("label")

df = pd.DataFrame(data, columns=columns)

df.to_csv(OUTPUT_CSV, index=False)

print("\n===================================")
print("CSV saved :", OUTPUT_CSV)
print("Total samples :", len(df))
print("Total columns :", len(df.columns))
print("===================================")