import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("datasetkata1.csv")

print("\n========================")
print("DATASET INFO")
print("========================")

print("Jumlah data :", len(df))
print("Jumlah kelas:", df["label"].nunique())

print("\nDistribusi kelas:")
print(df["label"].value_counts())

# =========================
# CLEAN DATA
# =========================
df = df.dropna()

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

print("\n========================")
print("MAPPING LABEL")
print("========================")

for i, label in enumerate(encoder.classes_):
    print(f"{i:02d} -> {label}")

# =========================
# TRAIN TEST SPLIT (FIX STRATIFY + BALANCE CHECK)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain :", len(X_train))
print("Test  :", len(X_test))

# =========================
# MODEL (OPTIMIZED FOR GESTURE)
# =========================
model = RandomForestClassifier(
    n_estimators=500,          # 🔥 lebih stabil dari 400
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",  # 🔥 lebih bagus dari balanced
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...\n")

model.fit(X_train, y_train)

# =========================
# EVALUASI
# =========================
pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n========================")
print("AKURASI :", round(acc * 100, 2), "%")
print("========================\n")

print("Classification Report:\n")
print(classification_report(y_test, pred, target_names=encoder.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))

# =========================
# ANALISIS BIAS (IMPORTANT)
# =========================
print("\n========================")
print("CHECK CLASS SUPPORT")
print("========================")

for i, label in enumerate(encoder.classes_):
    count = sum(y_test == i)
    print(f"{label}: {count}")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "model_kata.pkl")
joblib.dump(encoder, "encoder_kata.pkl")

print("\n========================")
print("MODEL BERHASIL DISIMPAN")
print("========================")