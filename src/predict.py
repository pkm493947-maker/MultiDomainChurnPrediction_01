# predict.py

import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model


MODEL_PATH = "../models/combined_ann.keras"
SCALER_PATH = "../models/combined_scaler.pkl"
FEATURE_PATH = "../models/combined_features.pkl"


def risk_level(prob):

    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_churn(input_file, output_file, threshold=0.5):

    print("\n----------------------------------------")
    print("📌 Predicting Churn for File:", input_file)

    df = pd.read_csv(input_file)

    print("✅ Input File Loaded")
    print("📊 Input Shape:", df.shape)

    # Remove target if exists
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    # Load model
    model = load_model(MODEL_PATH)
    print("✅ Model Loaded")

    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler Loaded")

    feature_list = joblib.load(FEATURE_PATH)
    print("✅ Training Feature List Loaded")

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Align columns
    print("📌 Aligning features...")
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_list]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict probability
    probs = model.predict(X_scaled).flatten()

    df["Churn_Probability"] = probs

    # Apply threshold
    df["Churn_Prediction"] = (probs >= threshold).astype(int)

    # Add Risk Level
    df["Risk_Level"] = df["Churn_Probability"].apply(risk_level)

    df.to_csv(output_file, index=False)

    print(f"✅ Prediction Saved Successfully: {output_file}")
    print("----------------------------------------\n")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":

    os.makedirs("../outputs", exist_ok=True)

    predict_churn(
        input_file="../outputs/selected_features.csv",
        output_file="../outputs/predicted_combined.csv",
        threshold=0.5
    )

    print("🎉 All Predictions Completed Successfully!")