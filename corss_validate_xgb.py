import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def evaluate_fold(y_true, y_pred):
    """Calculate metrics for a single fold"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1


def main():
    # 1. Read the preprocessed data
    df = pd.read_csv("tweets_preprocessed.csv", encoding="utf-8")
    X = df["text"]
    y = df["negativereason"]

    # 2. Encode target labels to numeric
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 3. Initialize models and cross-validation
    tfidf = TfidfVectorizer()
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 4. Initialize results storage
    results = []
    cm_sum = np.zeros((len(label_encoder.classes_), len(label_encoder.classes_)), dtype=int)

    # 5. Perform manual cross-validation
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Transform text data
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # Train and predict
        xgb.fit(X_train_tfidf, y_train)
        y_pred = xgb.predict(X_test_tfidf)

        # Calculate metrics
        accuracy, precision, recall, f1 = evaluate_fold(y_test, y_pred)
        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1
        })

        # Update confusion matrix
        cm_fold = confusion_matrix(y_test, y_pred)
        cm_sum += cm_fold

    # 6. Create results DataFrame
    df_cv = pd.DataFrame(results)

    # Add mean row
    mean_row = pd.DataFrame([{
        'fold': 'mean',
        'accuracy': df_cv['accuracy'].mean(),
        'precision_macro': df_cv['precision_macro'].mean(),
        'recall_macro': df_cv['recall_macro'].mean(),
        'f1_macro': df_cv['f1_macro'].mean()
    }])

    df_cv = pd.concat([df_cv, mean_row], ignore_index=True)

    # 7. Save results
    cv_csv_filename = "cv_results_xgb.csv"
    df_cv.to_csv(cv_csv_filename, index=False, encoding="utf-8")
    print(f"10-fold cross-validation (XGBoost) completed. Results saved to {cv_csv_filename}")
    print(df_cv)

    # 8. Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_sum, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Summed Over 10 Folds, XGBoost)")
    plt.colorbar()

    # Add labels
    labels = label_encoder.classes_
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)

    # Add text annotations
    threshold = cm_sum.max() / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm_sum[i, j]),
                     horizontalalignment="center",
                     color="white" if cm_sum[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("confusion_matrix_xgb.png", dpi=300, bbox_inches="tight")
    print("Confusion matrix saved as 'confusion_matrix_xgb.png'")


if __name__ == "__main__":
    main()