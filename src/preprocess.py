import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_combined(file_path, output_path):

    print("\n📌 Preprocessing Dataset...")

    # ✅ Correct loading
    df = pd.read_csv(file_path)

    print("✅ Dataset Loaded")
    print("Original Shape:", df.shape)

    # 🔥 Remove ID / leakage columns if exist
    drop_cols = ["CustomerID", "customerID", "ID"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    print("✅ ID / Leakage Columns Removed (if present)")

    # Separate target
    target = df["Churn"]
    df = df.drop(columns=["Churn"])

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    print("✅ Categorical Columns Encoded")

    # Scale features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    # Add target back
    df_scaled["Churn"] = target.values

    print("✅ Final Shape After Preprocessing:", df_scaled.shape)

    df_scaled.to_csv(output_path, index=False)

    print("✅ Preprocessed Dataset Saved:", output_path)
    print("----------------------------------------")

    return df_scaled