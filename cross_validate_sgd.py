import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def main():
    # 1. Read the preprocessed data
    df = pd.read_csv("tweets_preprocessed.csv", encoding="utf-8")
    # Assume it contains "text" (tweet content) and "negativereason" (target label)
    X = df["text"]
    y = df["negativereason"]

    # 2. Define a pipeline with TF-IDF + SGD
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("sgd", SGDClassifier(random_state=42))
    ])

    # 3. Cross-validation for metrics (using cross_validate)
    scoring_metrics = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=10,
        scoring=scoring_metrics,
        return_train_score=False
    )

    # 4. Store the results of each fold in a DataFrame
    df_cv = pd.DataFrame({
        "fold": range(1, 11),
        "accuracy": cv_results["test_accuracy"],
        "precision_macro": cv_results["test_precision_macro"],
        "recall_macro": cv_results["test_recall_macro"],
        "f1_macro": cv_results["test_f1_macro"]
    })

    # 5. Compute average across folds and append as a new row
    mean_row = {
        "fold": "mean",
        "accuracy": df_cv["accuracy"].mean(),
        "precision_macro": df_cv["precision_macro"].mean(),
        "recall_macro": df_cv["recall_macro"].mean(),
        "f1_macro": df_cv["f1_macro"].mean()
    }
    df_cv = pd.concat([df_cv, pd.DataFrame([mean_row])], ignore_index=True)

    # 6. Export cross-validation results to CSV
    cv_csv_filename = "cv_results_sgd.csv"
    df_cv.to_csv(cv_csv_filename, index=False, encoding="utf-8")
    print(f"10-fold cross-validation (SGD) completed. The results have been saved to {cv_csv_filename}\n")
    print(df_cv)

    # 7. Generate an aggregated confusion matrix by manually iterating over the folds.
    labels = sorted(df["negativereason"].unique())
    label_count = len(labels)
    cm_sum = np.zeros((label_count, label_count), dtype=int)

    # Splits the dataset into stratified folds
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # For each fold, train on (k-1) folds, predict on the remaining fold, then accumulate the confusion matrix
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cm_fold = confusion_matrix(y_test, y_pred, labels=labels)
        cm_sum += cm_fold

    # 8. Plot the aggregated confusion matrix as a colored PNG
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_sum, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Summed Over 10 Folds, SGD)")
    plt.colorbar()

    tick_marks = np.arange(label_count)
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    threshold = cm_sum.max() / 2.0

    for i in range(label_count):
        for j in range(label_count):
            plt.text(
                j,
                i,
                str(cm_sum[i, j]),
                horizontalalignment="center",
                color="white" if cm_sum[i, j] > threshold else "black"
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    plt.savefig("confusion_matrix_sgd.png", dpi=300, bbox_inches="tight")
    print("A colorful confusion matrix (SGD) has been saved as 'confusion_matrix_sgd.png'.")

if __name__ == "__main__":
    main()
