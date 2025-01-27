
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
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
    #    (b) Train an SVM classifier
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm", SVC(kernel="linear", probability=True, random_state=42))
    ])

    # 4. Fit the pipeline on the training data
    pipeline.fit(X, y)

    # 5. Save the trained pipeline (model) to a file
    #    We use joblib for serialization
    model_filename = "svm_model.pkl"
    joblib.dump(pipeline, model_filename)

    print(f"Model has been trained and saved to {model_filename}.")


if __name__ == "__main__":
    main()