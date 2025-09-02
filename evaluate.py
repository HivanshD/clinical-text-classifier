
from joblib import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from utils import load_csv, save_confusion_matrix
import os

def main():
    model = load("models/model.joblib")
    test = load_csv("data/test.csv")
    preds = model.predict(test["text"])
    acc = accuracy_score(test["label"], preds)
    pr, rc, f1, _ = precision_recall_fscore_support(test["label"], preds, average="binary")
    print(f"Test accuracy: {acc:.3f}")
    print(f"Precision: {pr:.3f}  Recall: {rc:.3f}  F1: {f1:.3f}")
    print("\nDetailed report:\n", classification_report(test["label"], preds, digits=3))

    cm = confusion_matrix(test["label"], preds)
    save_confusion_matrix(cm, classes=["no-risk","risk"], out_path="results/confusion_matrix.png")
    print("Saved results/confusion_matrix.png")

if __name__ == "__main__":
    main()
