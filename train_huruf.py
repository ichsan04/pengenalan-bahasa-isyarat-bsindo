import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==================================
# LOAD DATASET
# ==================================

DATASET_FILE = "datasethuruf.csv"

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {DATASET_FILE}"
    )

df = pd.read_csv(DATASET_FILE)

print("Total data awal :", len(df))

# ==================================
# LABEL
# ==================================

df["label"] = df["label"].astype(str)

huruf = [chr(i) for i in range(ord("A"), ord("Z") + 1)]

df = df[df["label"].isin(huruf)]

print("Total data huruf :", len(df))
print("Jumlah kelas :", df["label"].nunique())
print("Daftar kelas :", sorted(df["label"].unique()))

# ==================================
# FEATURE DAN LABEL
# ==================================

X = df.drop(columns=["label"])
y = df["label"]

X = X.apply(pd.to_numeric, errors="coerce")

valid_rows = ~X.isna().any(axis=1)

X = X[valid_rows]
y = y[valid_rows]

print("Data valid :", len(X))

# ==================================
# ENCODER
# ==================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

# ==================================
# SPLIT DATA
# ==================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

print("\nTrain :", len(X_train))
print("Test  :", len(X_test))

# ==================================
# SCALER
# ==================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================
# MODEL
# ==================================

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=True
)

print("\n========================")
print("TRAINING MODEL HURUF")
print("========================\n")

model.fit(X_train, y_train)

# ==================================
# EVALUASI
# ==================================

pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)

print("\n========================")
print("AKURASI :", round(acc * 100, 2), "%")
print("========================\n")

print(
    classification_report(
        y_test,
        pred,
        target_names=encoder.classes_
    )
)

# ==================================
# SAVE MODEL
# ==================================

joblib.dump(model, "model_huruf1.pkl")
joblib.dump(scaler, "scaler_huruf1.pkl")
joblib.dump(encoder, "encoder_huruf1.pkl")

print("\nModel berhasil disimpan:")
print("✓ model_huruf1.pkl")
print("✓ scaler_huruf1.pkl")
print("✓ encoder_huruf1.pkl")