import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("datasetangka.csv", low_memory=False)

df["label"] = df["label"].astype(str)

# Ambil hanya angka 0-9
angka_labels = [str(i) for i in range(10)]

df = df[df["label"].isin(angka_labels)]

print("Jumlah data:", len(df))
print("Kelas:", sorted(df["label"].unique()))

# =========================
# FEATURE & LABEL
# =========================

X = df.drop(columns=["label"])
y = df["label"]

X = X.apply(pd.to_numeric, errors="coerce")

valid_rows = ~X.isna().any(axis=1)

X = X[valid_rows]
y = y[valid_rows]

# =========================
# ENCODE LABEL
# =========================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =========================
# SCALER
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# MODEL
# =========================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    verbose=True
)

print("\nTraining Model Angka...\n")

model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n===================")
print("AKURASI:", round(acc * 100, 2), "%")
print("===================\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=encoder.classes_
    )
)

# =========================
# SAVE
# =========================

joblib.dump(model, "model_angka.pkl")
joblib.dump(scaler, "scaler_angka.pkl")
joblib.dump(encoder, "encoder_angka.pkl")

print("\nModel berhasil disimpan:")
print("model_angka.pkl")
print("scaler_angka.pkl")
print("encoder_angka.pkl")