import pandas as pd
from blockchain_storage import Blockchain
import os
import json

# Initialize blockchain
bc = Blockchain()

# List of retention files
files = [
    "outputs/retention_telecom.csv",
    "outputs/retention_banking.csv",
    "outputs/retention_ecommerce.csv"
]

# Columns possible names for auto-detection
possible_risk_cols = ["Risk_Level", "Risk", "Churn_Risk", "RiskCategory", "Risk_Category", "risk_level"]
possible_retention_cols = ["Retention_Action", "RetentionStrategy", "Retention_Strategy", "Action", "retention_action"]

for file in files:
    print(f"\n----------------------------------------")
    print(f"📌 Generating Alerts + Blockchain Logging for: {file}")

    # Extract domain name from filename
    domain = os.path.basename(file).split("_")[1].split(".")[0]  # telecom, banking, ecommerce

    # Load retention file
    df = pd.read_csv(file)
    print(f"✅ Input file loaded successfully! Shape: {df.shape}")

    # Auto-detect columns
    risk_col = next((col for col in possible_risk_cols if col in df.columns), None)
    retention_col = next((col for col in possible_retention_cols if col in df.columns), None)

    if not risk_col or not retention_col:
        print(f"❌ ERROR: Required columns not found! Available columns: {list(df.columns)}")
        continue

    # Generate alerts
    def alert_message(row):
        risk_value = str(row[risk_col]).lower()
        if risk_value in ["high", "critical"]:
            return f"⚠️ ALERT: Customer at {row[risk_col]} risk. Suggested Action: {row[retention_col]}"
        elif risk_value in ["medium"]:
            return f"⚠️ CAUTION: Customer at {row[risk_col]} risk. Suggested Action: {row[retention_col]}"
        else:
            return f"✅ Customer at {row[risk_col]} risk. No immediate action needed."

    df['Alert_Message'] = df.apply(alert_message, axis=1)

    # Save alerts CSV
    alert_file = file.replace("retention", "alerts")
    df.to_csv(alert_file, index=False)
    print(f"✅ Alerts Saved Successfully: {alert_file}")

    # Add each alert to blockchain
    for _, row in df.iterrows():
        transaction = {
            "CustomerID": row.get("RowNumber", row.get("id", "N/A")),
            "Risk_Level": row[risk_col],
            "Retention_Action": row[retention_col],
            "Alert_Message": row['Alert_Message']
        }
        bc.add_transaction(transaction)

    # Create a new block for this domain
    bc.create_block()
    blockchain_file = f"outputs/blockchain_{domain}.json"
    with open(blockchain_file, "w") as f:
        json.dump(bc.chain, f, indent=4)
    print(f"🎉 Blockchain Saved Successfully: {blockchain_file}")

print("\n----------------------------------------\n")
print("✅ All Alerts + Blockchain Logging Completed Successfully!")
