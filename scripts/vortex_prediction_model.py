import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

from vortexdetect.feature_processor import FeatureProcessor

# === Config ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "ml_ready_vortex_data.csv"
FEATURES_FILE = BASE_DIR / "data" / "ml_ready_features.csv"
MODEL_FILE = BASE_DIR / "models" / "xgb_vortex_model.joblib"
RESULTS_DIR = BASE_DIR / "results" / "vortex_xgb"
LOOKAHEAD = 10
THRESHOLD_STEP = 0.01


def prepare_data(df, feature_processor, force_recalculate=False):
    if FEATURES_FILE.exists() and not force_recalculate:
        print("📥 Loading precomputed features...")
        return pd.read_csv(FEATURES_FILE)

    print("⚙️ Recomputing features...")
    features = []

    for i in range(50, len(df) - LOOKAHEAD):
        window = df.iloc[i - 50:i]
        label = 1 if any(df["gt_detection_win"].iloc[i:i + LOOKAHEAD]) else 0
        feats = feature_processor.compute_features(window)
        feats["ml_label"] = label
        features.append(feats)

    result_df = pd.DataFrame(features)
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(FEATURES_FILE, index=False)
    return result_df


def train_model(X_train, y_train):
    print("🧠 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0, 1.01, THRESHOLD_STEP)
    best_f1, best_thresh = 0, 0
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1


def evaluate(model, X_test, y_test, save_dir):
    y_probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_test, y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_thresh
    }]).to_csv(save_dir / "model_metrics.csv", index=False)

    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_thresh
    }


def main(args):
    print("📄 Loading raw time-series data...")
    df = pd.read_csv(DATA_FILE)
    processor = FeatureProcessor()

    print("🔍 Preparing features and labels...")
    data = prepare_data(df, processor, force_recalculate=args.force_recalculate)
    data = data.dropna()
    X = data.drop(columns=["ml_label"])
    y = data["ml_label"]

    split = int(len(X) * args.data_fraction)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if MODEL_FILE.exists() and not args.force_retrain:
        print("📦 Loading existing model...")
        model = load(MODEL_FILE)
    else:
        model = train_model(X_train, y_train)
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        dump(model, MODEL_FILE)
        print(f"💾 Model saved to {MODEL_FILE}")

    print("📈 Evaluating model...")
    evaluate(model, X_test, y_test, RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost Vortex Prediction Model Runner")
    parser.add_argument("--force-recalculate", action="store_true", help="Recompute features from raw data")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain model from scratch")
    parser.add_argument("--data-fraction", type=float, default=0.8, help="Fraction of data to use for training")
    args = parser.parse_args()

    main(args)
