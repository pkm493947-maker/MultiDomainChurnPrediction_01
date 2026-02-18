import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model


def predict_churn(input_file, model_file, scaler_file, output_file, threshold=0.45):
    print("\n----------------------------------------")
    print(f"📌 Predicting Churn for File: {input_file}")

    # Load input dataset
    df = pd.read_csv(input_file)
    print("✅ Input file loaded successfully!")
    print("📌 Input Shape:", df.shape)

    # Load trained model
    model = load_model(model_file)
    print(f"✅ Model Loaded: {model_file}")

    # Load scaler
    scaler = joblib.load(scaler_file)
    print(f"✅ Scaler Loaded: {scaler_file}")

    # Remove target column if present
    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)
    if "Exited" in df.columns:
        df = df.drop("Exited", axis=1)

    # Handle missing values
    df = df.fillna(0)

    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)

    # Scale input features
    X_scaled = scaler.transform(df)

    # Predict probabilities
    y_prob = model.predict(X_scaled)

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Add predictions to dataframe
    df["Churn_Probability"] = y_prob
    df["Churn_Prediction"] = y_pred

    # Save output
    df.to_csv(output_file, index=False)
    print(f"✅ Prediction Saved Successfully: {output_file}")
    print("----------------------------------------\n")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    # Telecom Prediction
    predict_churn(
        input_file="outputs/selected_telecom.csv",
        model_file=r"C:\Users\Mamatha\Desktop\models\telecom_ann.keras",
        scaler_file=r"C:\Users\Mamatha\Desktop\models\telecom_ann_scaler.pkl",
        output_file="outputs/predicted_telecom.csv",
        threshold=0.45
    )

    # Banking Prediction
    predict_churn(
        input_file="outputs/selected_banking.csv",
        model_file=r"C:\Users\Mamatha\Desktop\models\banking_ann.keras",
        scaler_file=r"C:\Users\Mamatha\Desktop\models\banking_ann_scaler.pkl",
        output_file="outputs/predicted_banking.csv",
        threshold=0.45
    )

    # Ecommerce Prediction
    predict_churn(
        input_file="outputs/selected_ecommerce.csv",
        model_file=r"C:\Users\Mamatha\Desktop\models\ecommerce_ann.keras",
        scaler_file=r"C:\Users\Mamatha\Desktop\models\ecommerce_ann_scaler.pkl",
        output_file="outputs/predicted_ecommerce.csv",
        threshold=0.45
    )

    print("🎉 All Predictions Completed Successfully!")
