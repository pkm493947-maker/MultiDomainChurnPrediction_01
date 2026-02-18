import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(file_path, output_path, target_column=None):
    print("\n----------------------------------------")
    print(f"📌 Reading Dataset: {file_path}")

    # Auto detect separator (comma / semicolon / tab / etc.)
    df = pd.read_csv(file_path, sep=None, engine="python")

    print("✅ Dataset Loaded Successfully!")
    print("📌 Original Dataset Shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values
    df = df.ffill()

    # Encode categorical columns
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Scale numeric columns except target column
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if target_column is not None and target_column in numeric_cols:
        numeric_cols = numeric_cols.drop(target_column)

    if len(numeric_cols) > 0:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("✅ Cleaned Dataset Shape:", df.shape)

    # Save cleaned dataset
    df.to_csv(output_path, index=False)

    print(f"✅ Cleaned Dataset Saved: {output_path}")
    print("----------------------------------------")

    return df
