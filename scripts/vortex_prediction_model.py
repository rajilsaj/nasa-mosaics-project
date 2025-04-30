import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump, load
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
from vortexdetect.feature_processor import FeatureProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "ml_ready_vortex_data.csv"
FEATURE_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
THRESHOLD_STEP = 0.01
LOOKAHEAD = 10

def prepare_data(df, processor, window_size, features_file):
    if features_file.exists():
        print(f"Loading features for window {window_size}...")
        return pd.read_csv(features_file)

    print(f"Recomputing features for window {window_size}...")
    features = []
    for i in range(window_size, len(df) - LOOKAHEAD):
        window = df.iloc[i - window_size:i]
        label = 1 if df["gt_detection_win"].iloc[i:i + LOOKAHEAD].any() else 0
        feats = processor.compute_features(window)
        feats["ml_label"] = label
        features.append(feats)
    result_df = pd.DataFrame(features)
    result_df.to_csv(features_file, index=False)
    return result_df

def train_model(X, y):
    pos_weight = (len(y) - sum(y)) / sum(y)
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        random_state=42
    )
    model.fit(X, y)
    return model

def evaluate(model, X_test, y_test, window_size, result_path):
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0, 1.01, THRESHOLD_STEP)
    results = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        results.append((t, prec, rec, f1))

    best_t, best_p, best_r, best_f1 = max(results, key=lambda x: x[3])
    print(f"\nBest threshold based on F1: {best_t:.2f}")

    df_plot = pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1"])
    df_plot.to_csv(result_path / f"thresholds_window_{window_size}.csv", index=False)

    plt.figure()
    plt.plot(df_plot["Threshold"], df_plot["F1"], label="F1")
    plt.plot(df_plot["Threshold"], df_plot["Precision"], label="Precision")
    plt.plot(df_plot["Threshold"], df_plot["Recall"], label="Recall")
    plt.title(f"Threshold Tuning (Window {window_size})")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_path / f"threshold_plot_window_{window_size}.png")
    plt.close()

    return df_plot

def main(args):
    df = pd.read_csv(DATA_FILE)
    processor = FeatureProcessor()

    for window_size in args.windows:
        print(f"\n Window size: {window_size}")
        features_file = FEATURE_DIR / f"ml_ready_features_win_{window_size}.csv"
        data = prepare_data(df, processor, window_size, features_file).dropna()
        X = data.drop(columns=["ml_label"])
        y = data["ml_label"]

        split = int(len(X) * args.train_ratio)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = train_model(X_train, y_train)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        dump(model, MODEL_DIR / f"xgb_model_win_{window_size}.joblib")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        evaluate(model, X_test, y_test, window_size, RESULTS_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=[50, 60, 70, 80, 100])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
