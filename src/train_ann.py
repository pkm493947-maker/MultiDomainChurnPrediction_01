import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def train_ann_model(file_path, model_name, target_column="Churn", threshold=0.45):
    print("\n----------------------------------------")
    print(f"📌 Training ANN Model for: {file_path}")

    # Load dataset
    df = pd.read_csv(file_path)

    if target_column not in df.columns:
        print(f"❌ Target column '{target_column}' not found in dataset!")
        return

    # Features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only on training set
    print("✅ Applying SMOTE Oversampling ONLY on training data...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ANN Model
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation="relu"),
        Dropout(0.3),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(32, activation="relu"),
        Dropout(0.2),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Train
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Predict probabilities
    y_prob = model.predict(X_test)

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Results
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy of {model_name}: {acc:.4f}")
    print(f"🎯 Threshold used: {threshold}")

    print("\n📌 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n📌 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model + scaler
    model_folder = os.path.join(os.path.expanduser("~"), "Desktop", "models")
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(model_folder, f"{model_name}.keras")
    scaler_path = os.path.join(model_folder, f"{model_name}_scaler.pkl")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n✅ Model Saved: {model_path}")
    print(f"✅ Scaler Saved: {scaler_path}")
    print("----------------------------------------")


if __name__ == "__main__":
    train_ann_model("outputs/selected_telecom.csv", "telecom_ann", target_column="Churn", threshold=0.45)

    # Banking dataset target column is "Exited"
    train_ann_model("outputs/selected_banking.csv", "banking_ann", target_column="Exited", threshold=0.45)

    train_ann_model("outputs/selected_ecommerce.csv", "ecommerce_ann", target_column="Churn", threshold=0.45)

    print("\n🎉 All ANN Models Trained Successfully with SMOTE + EarlyStopping + Better Threshold!")
