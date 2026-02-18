import pandas as pd
import os


def assign_retention_strategy(prob):
    """
    Assigns risk category + retention action based on churn probability.
    """

    if prob >= 0.80:
        return "HIGH", "Call customer immediately + 30% discount + priority support"
    elif prob >= 0.60:
        return "MEDIUM", "Send email offer + 15% discount + loyalty bonus"
    elif prob >= 0.40:
        return "LOW", "Send SMS reminder + small coupon (5%)"
    else:
        return "SAFE", "No action needed (customer stable)"


def retention_plan(input_file, output_file):
    print("\n----------------------------------------")
    print(f"📌 Generating Retention Strategy for: {input_file}")

    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return

    df = pd.read_csv(input_file)

    # Possible prediction column names
    possible_columns = [
        "churn_probability",
        "Churn_Probability",
        "probability",
        "Probability",
        "predicted_probability",
        "Predicted_Probability",
        "ChurnProbability",
        "prediction",
        "Prediction"
    ]

    churn_prob_col = None

    for col in possible_columns:
        if col in df.columns:
            churn_prob_col = col
            break

    # If still not found, show columns
    if churn_prob_col is None:
        print("❌ ERROR: No churn probability column found in file!")
        print("📌 Available columns are:")
        print(df.columns)
        return

    print(f"✅ Using churn probability column: {churn_prob_col}")

    # Apply retention strategy
    risk_levels = []
    actions = []

    for prob in df[churn_prob_col]:
        risk, action = assign_retention_strategy(prob)
        risk_levels.append(risk)
        actions.append(action)

    df["risk_level"] = risk_levels
    df["retention_action"] = actions

    # Save output file
    df.to_csv(output_file, index=False)

    print(f"✅ Retention Strategy Saved Successfully: {output_file}")
    print("----------------------------------------")


if __name__ == "__main__":

    retention_plan("outputs/predicted_telecom.csv", "outputs/retention_telecom.csv")
    retention_plan("outputs/predicted_banking.csv", "outputs/retention_banking.csv")
    retention_plan("outputs/predicted_ecommerce.csv", "outputs/retention_ecommerce.csv")

    print("\n🎉 All Retention Strategies Generated Successfully!")
