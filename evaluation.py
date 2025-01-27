#!/usr/bin/env python3
# evaluation.py
# This script loads a trained model (e.g., svm_model.pkl or rf_model.pkl),
# evaluates it on the testing set (testing.csv), prints a confusion matrix
# in the console, and saves a colored confusion matrix as a PNG file.
# It also prints a variety of evaluation metrics (accuracy, precision, recall, F1).

import sys
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main(model_filename):
    # 1. Load the testing data
    df_test = pd.read_csv("testing.csv", encoding="utf-8")
    X_test = df_test["text"]
    y_test = df_test["negativereason"]

    # 2. Load the trained model
    model = joblib.load(model_filename)

    # 3. Make predictions
    y_pred = model.predict(X_test)

    # 4. Compute evaluation metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Individual summaries per class
    report = classification_report(y_test, y_pred, zero_division=0)

    # 5. Print results to console
    print(f"Model File: {model_filename}")
    print("\n=== Confusion Matrix ===")
    print(cm)

    print("\n=== Classification Report ===")
    print(report)

    print("=== Overall Metrics (Macro-Averaged) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision_macro:.4f}")
    print(f"Recall:    {recall_macro:.4f}")
    print(f"F1 Score:  {f1_macro:.4f}")

    # 6. Create and save a colored confusion matrix as PNG
    labels = sorted(df_test["negativereason"].unique())  # sorted list of unique categories

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    # Add tick marks and labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels, rotation=0)

    # Add text labels inside the cells
    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cell_text = cm[i, j]
            color = "white" if cell_text > threshold else "black"
            plt.text(j, i, format(cell_text),
                     horizontalalignment="center",
                     color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Save as PNG
    output_fname = "colored_confusion_matrix.png"
    plt.savefig(output_fname, dpi=300, bbox_inches="tight")
    print(f"\nColored confusion matrix saved as '{output_fname}'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluation.py [model_filename.pkl]")
        sys.exit(1)
    model_filename = sys.argv[1]
    main(model_filename)