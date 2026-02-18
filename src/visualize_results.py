# visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import os

# List of domains
domains = ["telecom", "banking", "ecommerce"]

# Output folder paths
outputs_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
plots_folder = os.path.join(outputs_folder, "plots")
os.makedirs(plots_folder, exist_ok=True)

# Possible risk column names
possible_risk_cols = ["Risk_Level", "Risk", "Churn_Risk", "RiskCategory", "Risk_Category", "risk_level"]

for domain in domains:
    alert_file = os.path.join(outputs_folder, f"alerts_{domain}.csv")

    if not os.path.exists(alert_file):
        print(f"❌ Alerts file not found: {alert_file}")
        continue

    # Load alerts CSV
    df = pd.read_csv(alert_file)

    # Auto-detect risk column
    risk_col = next((col for col in possible_risk_cols if col in df.columns), None)
    if not risk_col:
        print(f"❌ Risk column not found in {alert_file}. Available columns: {list(df.columns)}")
        continue

    # -------------------------
    # 1️⃣ Pie Chart: Risk Distribution
    # -------------------------
    risk_counts = df[risk_col].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%", startangle=140, colors=["green", "orange", "red"])
    plt.title(f"{domain.capitalize()} Customers Risk Distribution")
    pie_file = os.path.join(plots_folder, f"{domain}_risk_distribution.png")
    plt.savefig(pie_file)
    plt.close()
    print(f"✅ Pie chart saved: {pie_file}")

    # -------------------------
    # 2️⃣ Bar Chart: Alerts Generated vs Total Customers
    # -------------------------
    total_customers = df.shape[0]
    high_risk = df[risk_col].str.lower().isin(["high", "critical"]).sum()
    medium_risk = df[risk_col].str.lower().isin(["medium"]).sum()
    low_risk = df[risk_col].str.lower().isin(["low"]).sum()
    alerts_generated = high_risk + medium_risk  # considering high + medium risk as alerts

    plt.figure(figsize=(6, 4))
    plt.bar(["Alerts Generated", "Total Customers"], [alerts_generated, total_customers], color=["red", "blue"])
    plt.title(f"{domain.capitalize()} Alerts vs Total Customers")
    plt.ylabel("Number of Customers")
    bar_file = os.path.join(plots_folder, f"{domain}_alerts_vs_total.png")
    plt.savefig(bar_file)
    plt.close()
    print(f"✅ Bar chart saved: {bar_file}")

print("\n🎉 All visualizations generated successfully!")
