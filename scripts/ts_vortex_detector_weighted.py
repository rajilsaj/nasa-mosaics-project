import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "address.csv"
RESULTS_DIR = BASE_DIR / "results" / "random_forest"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "mean_pressure", "std_pressure", "pressure_change", "pressure_drop_ratio", 
    "z_score", "scheme1_detection", "scheme2_detection"
]
TARGET = "ml_label"


def load_data(path):
    df = pd.read_csv(path)
    df["scheme1_detection"] = df["scheme1_detection"].astype(int)
    df["scheme2_detection"] = df["scheme2_detection"].astype(int)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return X, y


def split_data(X, y, ratio=0.8):
    split_idx = int(len(X) * ratio)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1_score": f1_score(y_test, preds, zero_division=0),
    }

    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return metrics, probs, preds


def tune_threshold(y_test, y_probs):
    thresholds = np.linspace(0, 1, 101)
    best_f1, best_thresh = 0, 0
    f1s, precisions, recalls = [], [], []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_test, preds, zero_division=0)
        f1s.append(f1)
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh, best_f1, thresholds, f1s, precisions, recalls


def plot_threshold_metrics(thresholds, f1s, precisions, recalls, best_thresh):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1s, label="F1 Score", marker="o")
    plt.plot(thresholds, precisions, label="Precision", marker="x")
    plt.plot(thresholds, recalls, label="Recall", marker="s")
    plt.axvline(best_thresh, linestyle="--", color="gray", label=f"Best: {best_thresh:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Tuning (Random Forest)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "threshold_tuning_rf.png")
    plt.close()


def save_metrics(metrics):
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)


def main():
    print("Loading and preparing data...")
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Evaluating at default threshold = 0.5...")
    base_metrics, y_probs, _ = evaluate(model, X_test, y_test)

    print("Tuning threshold for best F1...")
    best_thresh, best_f1, thresholds, f1s, precisions, recalls = tune_threshold(y_test, y_probs)
    print(f"\nBest Threshold: {best_thresh:.2f}, F1 Score: {best_f1:.3f}")

    plot_threshold_metrics(thresholds, f1s, precisions, recalls, best_thresh)
    save_metrics(base_metrics)


if __name__ == "__main__":
    main()
