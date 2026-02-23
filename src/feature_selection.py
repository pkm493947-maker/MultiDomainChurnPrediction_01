import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def select_features(file_path, output_path, top_k=25):

    print("\n🔍 Feature Selection Started...")

    df = pd.read_csv(file_path, low_memory=False)

    print("📊 Dataset Shape:", df.shape)

    if "Churn" not in df.columns:
        raise Exception("❌ Churn column missing!")

    # ✅ Convert Churn to numeric (VERY IMPORTANT FIX)
    df["Churn"] = pd.to_numeric(df["Churn"], errors="coerce")

    # Fill missing after conversion
    df = df.dropna(subset=["Churn"])

    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # ✅ Convert ALL columns to numeric (force safe conversion)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Fill NaN created by conversion
    X = X.fillna(0)

    print("⚡ Calculating Mutual Information...")

    mi_scores = mutual_info_classif(X, y)

    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "MI_Score": mi_scores
    })

    mi_df = mi_df.sort_values(by="MI_Score", ascending=False)

    print("\n🔝 Top Features:")
    print(mi_df.head(top_k))

    selected_features = mi_df["Feature"].head(top_k).tolist()

    X_selected = X[selected_features]
    X_selected["Churn"] = y.values

    X_selected.to_csv(output_path, index=False)

    print("✅ Selected Feature Dataset Saved:", output_path)
    print("------------------------------------")

    return selected_features