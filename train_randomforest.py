
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


def main():
    # 1. Read the training data
    df_train = pd.read_csv("training.csv", encoding="utf-8")

    # 2. Separate features (X) and target (y)
    X = df_train["text"]
    y = df_train["negativereason"]

    # 3. Create a pipeline that does:
    #    (a) TF-IDF vectorization on the text
    #    (b) Train a Random Forest classifier
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Fit the pipeline on the training data
    pipeline.fit(X, y)

    # 5. Save the trained pipeline (model) to a file
    model_filename = "rf_model.pkl"
    joblib.dump(pipeline, model_filename)

    print(f"Random Forest model has been trained and saved to {model_filename}.")


if __name__ == "__main__":
    main()