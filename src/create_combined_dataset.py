import pandas as pd
import os


def load_dataset_correctly(path):

    # 🔥 Important: Your files are TAB separated
    return pd.read_csv(path, sep="\t")


def create_combined_dataset():

    print("📌 Loading datasets correctly with TAB separator...")

    telecom = load_dataset_correctly("../datasets/telecom.csv")
    banking = load_dataset_correctly("../datasets/banking.csv")
    ecommerce = load_dataset_correctly("../datasets/ecommerce.csv")

    # ---------------- Normalize Target Column ----------------

    # Telecom
    if "churn" in telecom.columns:
        telecom.rename(columns={"churn": "Churn"}, inplace=True)

    # Banking
    if "Exited" in banking.columns:
        banking.rename(columns={"Exited": "Churn"}, inplace=True)

    # Ecommerce
    if "Churn" not in ecommerce.columns and "churn" in ecommerce.columns:
        ecommerce.rename(columns={"churn": "Churn"}, inplace=True)

    # Add domain column
    telecom["Domain"] = "Telecom"
    banking["Domain"] = "Banking"
    ecommerce["Domain"] = "Ecommerce"

    # Combine
    combined = pd.concat([telecom, banking, ecommerce], ignore_index=True)

    if "Churn" not in combined.columns:
        raise Exception("❌ Churn column STILL missing!")

    save_path = "../datasets/combined_data.csv"
    combined.to_csv(save_path, index=False)

    print("✅ Combined Dataset Created Successfully!")
    print("📊 Shape:", combined.shape)
    print("📁 Saved At:", save_path)


if __name__ == "__main__":
    create_combined_dataset()