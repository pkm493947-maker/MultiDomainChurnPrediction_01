import os
from preprocess import preprocess_data
from feature_selection import feature_selection
from train_ann import train_ann_model


# Get current file location (src folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root folder
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Dataset and output paths
DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Create folders if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

print("\n📂 Checking datasets folder...")
print("Files inside datasets folder:", os.listdir(DATASET_DIR))


# -------------------- STEP 1: PREPROCESSING --------------------

telecom_clean = os.path.join(OUTPUT_DIR, "cleaned_telecom.csv")
banking_clean = os.path.join(OUTPUT_DIR, "cleaned_banking.csv")
ecommerce_clean = os.path.join(OUTPUT_DIR, "cleaned_ecommerce.csv")

preprocess_data(os.path.join(DATASET_DIR, "telecom.csv"), telecom_clean, target_column="Churn")
preprocess_data(os.path.join(DATASET_DIR, "banking.csv"), banking_clean, target_column="Exited")
preprocess_data(os.path.join(DATASET_DIR, "ecommerce.csv"), ecommerce_clean, target_column="Churn")

print("\n🎉 Preprocessing Completed Successfully!")


# -------------------- STEP 2: FEATURE SELECTION --------------------

selected_telecom = os.path.join(OUTPUT_DIR, "selected_telecom.csv")
selected_banking = os.path.join(OUTPUT_DIR, "selected_banking.csv")
selected_ecommerce = os.path.join(OUTPUT_DIR, "selected_ecommerce.csv")

feature_selection(
    telecom_clean,
    target_column="Churn",
    output_file=selected_telecom,
    k=10
)

feature_selection(
    banking_clean,
    target_column="Exited",
    output_file=selected_banking,
    k=10
)

feature_selection(
    ecommerce_clean,
    target_column="Churn",
    output_file=selected_ecommerce,
    k=10
)

print("\n🎉 Feature Selection Completed Successfully!")


# -------------------- STEP 3: ANN TRAINING --------------------

print("\n🚀 Training ANN Models Started...")

train_ann_model(
    selected_telecom,
    target_column="Churn",
    model_name="telecom_ann",
    models_folder=MODELS_DIR
)

train_ann_model(
    selected_banking,
    target_column="Exited",
    model_name="banking_ann",
    models_folder=MODELS_DIR
)

train_ann_model(
    selected_ecommerce,
    target_column="Churn",
    model_name="ecommerce_ann",
    models_folder=MODELS_DIR
)

print("\n🎉 All ANN Models Trained Successfully!")
print("📂 Check models folder for saved ANN models.")
