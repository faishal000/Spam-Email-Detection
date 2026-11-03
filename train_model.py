"""Train and evaluate the Email Spam Detection model."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "emails.csv"
MODEL_FILE = BASE_DIR / "spam_model.joblib"
METRICS_FILE = BASE_DIR / "model_metrics.json"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError("emails.csv must contain 'text' and 'label' columns.")

    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["text"] != ""]
    data = data[data["label"].isin(["spam", "ham"])]

    if data["label"].nunique() != 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Dataset must contain at least two examples of both spam and ham.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25,
        random_state=42, stratify=data["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB(alpha=0.5))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = ["ham", "spam"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_spam": float(precision_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_test, predictions, pos_label="spam", zero_division=0)),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, target_names=["Not Spam (Ham)", "Spam"],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "dataset_size": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in data["label"].value_counts().items()},
        "note": "Educational sample dataset; metrics are illustrative and not a real-world guarantee."
    }

    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nEMAIL SPAM DETECTION - MODEL EVALUATION")
    print("=" * 42)
    print(f"Dataset rows : {metrics['dataset_size']}")
    print(f"Training rows: {metrics['train_size']}")
    print(f"Testing rows : {metrics['test_size']}")
    print(f"Accuracy     : {metrics['accuracy']:.2%}")
    print(f"Spam precision: {metrics['precision_spam']:.2%}")
    print(f"Spam recall   : {metrics['recall_spam']:.2%}")
    print(f"Spam F1-score : {metrics['f1_spam']:.2%}")
    print("\nConfusion matrix (rows=actual [ham, spam], columns=predicted [ham, spam]):")
    print(cm)
    print("\nSaved model:", MODEL_FILE.name)
    print("Saved metrics:", METRICS_FILE.name)
    print("\nReminder: the included 50-message dataset is a classroom demo. Use a larger, independently sourced dataset for meaningful performance claims.")


if __name__ == "__main__":
    main()
