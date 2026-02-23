import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf


def visualize_results(model_path, dataset_path, target_column="Churn", output_folder="../outputs"):
    print("\n📊 Generating Visual Reports...")

    # Load trained model
    model = tf.keras.models.load_model(model_path)

    # Load dataset
    df = pd.read_csv(dataset_path)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Get predictions
    y_prob = model.predict(X)
    y_pred = (y_prob > 0.5).astype(int)

    # ==========================
    # 1️⃣ Confusion Matrix
    # ==========================
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_folder, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print("✅ Confusion Matrix Saved:", cm_path)

    # ==========================
    # 2️⃣ Prediction Distribution
    # ==========================
    plt.figure(figsize=(6, 5))
    plt.hist(y_prob, bins=50)
    plt.title("Prediction Probability Distribution")

    hist_path = os.path.join(output_folder, "prediction_distribution.png")
    plt.savefig(hist_path)
    plt.close()

    print("✅ Prediction Distribution Saved:", hist_path)

    print("✅ Visualization Completed")