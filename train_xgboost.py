import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

# === Try to import imblearn (for balancing)
try:
    from imblearn.under_sampling import RandomUnderSampler
    USE_IMBLEARN = True
except ImportError:
    print("⚠️ imblearn not installed. Proceeding without class balancing.")
    USE_IMBLEARN = False

# === Paths ===
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_PATH = "data/ml_ready_vortex_data.csv"
MODEL_PATH = f"{RESULTS_DIR}/xgboost_vortex_model.pkl"
THRESHOLD_PLOT_PATH = f"{RESULTS_DIR}/threshold_tuning.png"
CONFUSION_MATRIX_PLOT_PATH = f"{RESULTS_DIR}/confusion_matrix.png"
METRICS_PATH = f"{RESULTS_DIR}/report.txt"
BEST_THRESHOLD_PATH = f"{RESULTS_DIR}/best_threshold.txt"

# === Preprocessing ===
def preprocess(df):
    df['pressure_diff'] = df['PRESSURE'] - df['PRESSURE_MA_500']
    df['pressure_diff_percent'] = (df['pressure_diff'] / df['PRESSURE_MA_500']) * 100

    for window in [10, 25]:
        df[f'rolling_mean_{window}'] = df['pressure_diff'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['pressure_diff'].rolling(window).std()

    for lag in [1, 2]:
        df[f'pressure_lag_{lag}'] = df['PRESSURE'].shift(lag)

    # Slope over window
    window = 25
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    def slope(series):
        if np.isnan(series).any():
            return np.nan
        return ((x - x_mean) * (series - series.mean())).sum() / denom
    df['slope_25'] = df['PRESSURE'].rolling(window=window).apply(slope, raw=True)

    df['drop_1_percent'] = (df['pressure_diff_percent'] < -1.0).astype(int)
    df['count_drops_25'] = df['drop_1_percent'].rolling(window=25).sum()

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df['hour_fraction'] = df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600
    df['sin_time'] = np.sin(2 * np.pi * df['hour_fraction'] / 24.0)
    df['cos_time'] = np.cos(2 * np.pi * df['hour_fraction'] / 24.0)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    return df

# === Train & Evaluate ===
def train_model(X, y):
    # Handle imbalance
    if USE_IMBLEARN:
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y)
    else:
        print("⚠️ Class imbalance not handled! Results may be biased.")

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Save report
    with open(METRICS_PATH, "w") as f:
        f.write("Classification Report:\n" + report + "\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n")
        f.write("ROC-AUC Score:\n" + str(auc) + "\n")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["False", "True"], yticklabels=["False", "True"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close()

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]

    with open(BEST_THRESHOLD_PATH, "w") as f:
        f.write(str(best_threshold))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', marker='o')
    plt.plot(thresholds, precision[:-1], label='Precision', linestyle='--')
    plt.plot(thresholds, recall[:-1], label='Recall', linestyle='--')
    plt.axvline(x=best_threshold, color='gray', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
    plt.title("Threshold Tuning")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(THRESHOLD_PLOT_PATH)
    plt.close()

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"📉 ROC-AUC Score: {auc:.4f}")
    print(f"🎯 Best Threshold: {best_threshold:.4f}")

# === Main ===
def main():
    print("🚀 Loading data...")
    df = pd.read_csv(DATA_PATH)
    df = preprocess(df)

    features = [
        'PRESSURE', 'PRESSURE_MA_500', 'pressure_diff', 'pressure_diff_percent',
        'rolling_mean_10', 'rolling_std_10', 'rolling_mean_25', 'rolling_std_25',
        'pressure_lag_1', 'pressure_lag_2', 'slope_25',
        'count_drops_25', 'sin_time', 'cos_time'
    ]
    X = df[features]
    y = df['gt_4xfwhm']

    if y.nunique() < 2:
        print("❌ Dataset contains only one class. Aborting.")
        return

    train_model(X, y)

if __name__ == "__main__":
    main()
