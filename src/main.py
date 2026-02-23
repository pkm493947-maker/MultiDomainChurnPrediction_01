import os
import pandas as pd

from preprocess import preprocess_combined
from feature_selection import select_features
from train_ann import train_ann_model
from blockchain_storage import store_blockchain_record
from visualize_results import visualize_results


# ==========================================================
# 🔥 PATH SETUP
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("\n📂 Project Initialized")
print("Datasets Folder:", DATASET_DIR)


# ==========================================================
# 🔥 STEP 1 — LOAD COMBINED DATASET
# ==========================================================

combined_path = os.path.join(DATASET_DIR, "combined_data.csv")

if not os.path.exists(combined_path):
    print("❌ combined_data.csv NOT FOUND!")
    print("Run create_combined_dataset.py first.")
    exit()

print("\n✅ Combined Dataset Found:", combined_path)


# ==========================================================
# 🔥 STEP 2 — PREPROCESS
# ==========================================================

processed_path = os.path.join(OUTPUT_DIR, "preprocessed_combined.csv")

preprocess_combined(
    file_path=combined_path,
    output_path=processed_path
)


# ==========================================================
# 🔥 STEP 3 — FEATURE SELECTION
# ==========================================================

selected_path = os.path.join(OUTPUT_DIR, "selected_features.csv")

select_features(
    file_path=processed_path,
    output_path=selected_path,
    top_k=25
)


# ==========================================================
# 🔥 STEP 4 — TRAIN ANN MODEL (WITH SMOTE + AUTO THRESHOLD)
# ==========================================================

model = train_ann_model(
    data_path=selected_path,     # ✅ change here
    target_column="Churn",
    model_name="multi_domain_ann",
    models_folder=MODELS_DIR
)


# ==========================================================
# 🔥 STEP 5 — STORE MODEL RESULT IN BLOCKCHAIN
# ==========================================================

print("\n🔗 Storing Result in Blockchain...")

blockchain_data = {
    "model": "Multi Domain ANN",
    "dataset": "Combined Telecom + Banking + Ecommerce",
    "status": "Trained Successfully"
}

store_blockchain_record(blockchain_data)

print("✅ Blockchain Record Stored")


# ==========================================================
# 🔥 STEP 6 — VISUALIZE RESULTS
# ==========================================================

print("\n📊 Generating Visual Reports...")

visualize_results(
    model_path=os.path.join(MODELS_DIR, "multi_domain_ann.keras"),
    dataset_path=selected_path
)

print("\n🎉 PROJECT EXECUTION COMPLETED SUCCESSFULLY 🚀")
print("Check models / outputs / blockchain folders.")