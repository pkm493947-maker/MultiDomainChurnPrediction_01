# train_ann.py
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers


# ==============================
# CONFIG
# ==============================

DATA_PATH = "../outputs/selected_features.csv"  # Combined selected dataset
MODEL_SAVE_PATH = "../models/combined_ann.keras"
SCALER_SAVE_PATH = "../models/combined_scaler.pkl"
FEATURE_SAVE_PATH = "../models/combined_features.pkl"


# ==============================
# TRAIN FUNCTION
# ==============================

def train_ann_model():

    print("\n------------------------------------")
    print("📌 Loading Dataset for Training")

    df = pd.read_csv(DATA_PATH)

    print("✅ Dataset Loaded")
    print("📊 Shape:", df.shape)

    # Separate target
    if "Churn" not in df.columns:
        raise ValueError("❌ Churn column missing!")

    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✅ Data Split Completed")

    # -----------------------
    # SMOTE
    # -----------------------

    print("⚖ Applying SMOTE...")

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("✅ SMOTE Applied")
    print(y_train.value_counts())

    # -----------------------
    # Scaling
    # -----------------------

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("✅ Scaler Saved")

    # Save feature list for prediction alignment
    feature_list = X.columns.tolist()
    joblib.dump(feature_list, FEATURE_SAVE_PATH)
    print("✅ Feature List Saved:", FEATURE_SAVE_PATH)

    # -----------------------
    # ANN MODEL
    # -----------------------

    print("🚀 Training Model...")

    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32
    )

    # Save Model
    model.save(MODEL_SAVE_PATH)

    print("💾 Model Saved:", MODEL_SAVE_PATH)
    print("------------------------------------")
    print("🎉 Training Finished Successfully!")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    os.makedirs("../models", exist_ok=True)
    train_ann_model()