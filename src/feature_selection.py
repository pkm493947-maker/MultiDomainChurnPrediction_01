import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


def feature_selection(input_file, target_column, output_file, k=10):
    print("\n----------------------------------------")
    print(f"📌 Feature Selection for: {input_file}")

    df = pd.read_csv(input_file)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # chi-square requires non-negative values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_scaled, y)

    selected_features = X.columns[selector.get_support()]

    print("✅ Selected Features:", list(selected_features))

    new_df = df[selected_features].copy()
    new_df[target_column] = y

    new_df.to_csv(output_file, index=False)

    print(f"✅ Feature Selected Dataset Saved: {output_file}")
    print("----------------------------------------")

    return list(selected_features)
