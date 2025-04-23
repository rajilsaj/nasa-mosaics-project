import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

from vortexdetect.feature_processor import FeatureProcessor

# === Config ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "ml_ready_vortex_data.csv"
FEATURES_FILE = BASE_DIR / "data" / "ml_ready_features.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
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


def train_model(X_train, y_train, model_type="xgboost"):
    print(f"🧠 Training {model_type.upper()} model...")

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    else:
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


def evaluate(model, X_test, y_test, save_dir, model_type):
    y_probs = model.predict_proba(X_test)[:, 1]
    best_thresh, best_f1 = find_best_threshold(y_test, y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    result_path = save_dir / f"{model_type}"
    result_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_thresh
    }]).to_csv(result_path / "model_metrics.csv", index=False)

    with open(result_path / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": best_thresh
    }


def main(args):
    model_type = args.model.lower()
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

    model_file = MODEL_DIR / f"{model_type}_vortex_model.joblib"

    if model_file.exists() and not args.force_retrain:
        print("📦 Loading existing model...")
        model = load(model_file)
    else:
        model = train_model(X_train, y_train, model_type=model_type)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dump(model, model_file)
        print(f"💾 Model saved to {model_file}")

    print("📈 Evaluating model...")
    evaluate(model, X_test, y_test, RESULTS_DIR, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vortex Prediction Model Trainer")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "rf"],
                        help="Choose model type: 'xgboost' or 'rf'")
    parser.add_argument("--force-recalculate", action="store_true", help="Recompute features from raw data")
    parser.add_argument("--force-retrain", action="store_true", help="Retrain model from scratch")
    parser.add_argument("--data-fraction", type=float, default=0.8, help="Fraction of data to use for training")
    args = parser.parse_args()

    main(args)
