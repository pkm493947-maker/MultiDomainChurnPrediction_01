import os
import numpy as np
import pandas as pd
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
import joblib

# =====================================================
# 🔥 PROJECT PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "combined_ann.keras")
SCALER_PATH = os.path.join(BASE_DIR, "models", "combined_scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "combined_features.pkl")
BLOCKCHAIN_PATH = os.path.join(BASE_DIR, "blockchain", "ledger.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(BLOCKCHAIN_PATH), exist_ok=True)

# =====================================================
# 🔥 LOAD MODEL SAFELY
# =====================================================

try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURE_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Loading Failed:", e)
    model = None
    scaler = None
    features = []

# =====================================================
# 🔥 FLASK APP
# =====================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =====================================================
# 🔥 RISK CLASSIFICATION
# =====================================================

def get_risk_level(prob):
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"

# =====================================================
# 🔥 PERSONALIZED RETENTION STRATEGY
# =====================================================

def get_retention_strategy(prob, row):
    if prob >= 0.7:
        if hasattr(row, "MonthlyCharges") and row.MonthlyCharges > 80:
            return "Offer 25% Discount + Dedicated Support Call"
        elif hasattr(row, "Contract") and row.Contract == "Month-to-month":
            return "Offer 1-Year Contract Upgrade with Discount"
        else:
            return "Immediate Retention Team Intervention"

    elif prob >= 0.4:
        return "Send Loyalty Points + Targeted Promotion Email"

    else:
        return "Upsell Premium Plan + Appreciation Offer"

# =====================================================
# 🔥 BLOCKCHAIN FUNCTIONS
# =====================================================

def load_blockchain():
    if os.path.exists(BLOCKCHAIN_PATH):
        with open(BLOCKCHAIN_PATH, "r") as f:
            return json.load(f)
    return []

def save_blockchain(data):
    with open(BLOCKCHAIN_PATH, "w") as f:
        json.dump(data, f, indent=4)

# =====================================================
# 🔥 LANDING PAGE
# =====================================================

@app.route("/")
def landing():
    return render_template("landing.html")

# =====================================================
# 🔥 DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():
    blockchain = load_blockchain()

    if len(blockchain) > 0:
        latest = blockchain[-1]
    else:
        latest = {
            "total_customers": 0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "timestamp": "-"
        }

    return render_template(
        "dashboard.html",
        total_uploads=len(blockchain),
        total_customers=latest["total_customers"],
        high_risk=latest["high_risk"],
        medium_risk=latest["medium_risk"],
        low_risk=latest["low_risk"],
        last_updated=latest["timestamp"]
    )

# =====================================================
# 🔥 UPLOAD PAGE
# =====================================================

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

# =====================================================
# 🔥 PREDICTION ENGINE
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return "Model not loaded properly."

    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]

    if file.filename == "":
        return "No selected file."

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    df_original = pd.read_csv(filepath)
    df = df_original.copy()

    # Feature Engineering
    df = df.fillna(0)
    df = pd.get_dummies(df)

    # Align with training features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    probs = model.predict(X_scaled).flatten()
    predictions = (probs >= 0.5).astype(int)

    # Add Results
    df_original["Probability"] = probs
    df_original["Prediction"] = predictions
    df_original["Risk"] = [get_risk_level(p) for p in probs]
    df_original["Strategy"] = [
        get_retention_strategy(p, row)
        for p, row in zip(probs, df_original.itertuples())
    ]

    # Save Result File
    result_file = "prediction_result.csv"
    result_path = os.path.join(UPLOAD_FOLDER, result_file)
    df_original.to_csv(result_path, index=False)

    # High Risk Customers
    high_risk_df = df_original[df_original["Risk"] == "High"]

    # Blockchain Logging
    blockchain = load_blockchain()

    record = {
        "file": file.filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_customers": len(df_original),
        "high_risk": int(sum(df_original["Risk"] == "High")),
        "medium_risk": int(sum(df_original["Risk"] == "Medium")),
        "low_risk": int(sum(df_original["Risk"] == "Low")),
        "high_risk_ids": high_risk_df.iloc[:, 0].tolist()
    }

    blockchain.append(record)
    save_blockchain(blockchain)

    return render_template(
        "results.html",
        customers=df_original.to_dict(orient="records"),
        high_risk=high_risk_df.to_dict(orient="records"),
        result_file=result_file
    )

# =====================================================
# 🔥 ALERT PAGE
# =====================================================

@app.route("/alerts")
def alerts_page():
    blockchain = load_blockchain()

    if len(blockchain) > 0:
        latest = blockchain[-1]
        high_risk_ids = latest.get("high_risk_ids", [])
    else:
        high_risk_ids = []

    return render_template("alerts.html", high_risk_ids=high_risk_ids)

# =====================================================
# 🔥 BLOCKCHAIN PAGE
# =====================================================

@app.route("/blockchain")
def blockchain_page():
    blockchain = load_blockchain()
    return render_template("blockchain.html", blockchain=blockchain)

# =====================================================
# 🔥 RETENTION PAGE
# =====================================================

@app.route("/retention")
def retention_page():
    return render_template("retention.html")

# =====================================================
# 🔥 DOWNLOAD RESULT
# =====================================================

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename,
        as_attachment=True
    )

# =====================================================
# 🔥 RUN SERVER
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)