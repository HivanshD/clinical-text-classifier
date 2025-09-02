
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from utils import load_csv
import os

def main():
    train = load_csv("data/train.csv")
    val = load_csv("data/val.csv")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200))
    ])
    pipe.fit(train["text"], train["label"])
    preds = pipe.predict(val["text"])
    print("Validation accuracy:", round(accuracy_score(val["label"], preds), 3))
    print(classification_report(val["label"], preds, digits=3))
    os.makedirs("models", exist_ok=True)
    dump(pipe, "models/model.joblib")
    print("Saved models/model.joblib")

if __name__ == "__main__":
    main()
