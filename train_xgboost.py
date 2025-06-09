import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ==== SETUP ====
DATA_PATH = "data/ml_ready_vortex_data.csv"
MODEL_PATH = "results/xgboost_vortex_model.pkl"
THRESHOLD_PLOT_PATH = "results/threshold_tuning.png"
os.makedirs("results", exist_ok=True)


# ==== FUNCTION: Feature Engineering ====
def preprocess(df):
    # Base features
    df['pressure_diff'] = df['PRESSURE'] - df['PRESSURE_MA_500']
    df['pressure_diff_percent'] = (df['pressure_diff'] / df['PRESSURE_MA_500']) * 100

    for window in [10, 25]:
        df[f'rolling_mean_{window}'] = df['pressure_diff'].rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['pressure_diff'].rolling(window=window).std()

    for lag in [1, 2]:
        df[f'pressure_lag_{lag}'] = df['PRESSURE'].shift(lag)

    # Slope feature
    window = 25
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def fast_slope(series):
        if np.isnan(series).any():
            return np.nan
        return ((x - x_mean) * (series - series.mean())).sum() / denom

    df['slope_25'] = df['PRESSURE'].rolling(window=window).apply(fast_slope, raw=True)

    # Drop count
    df['drop_1_percent'] = (df['pressure_diff_percent'] < -1.0).astype(int)
    df['count_drops_25'] = df['drop_1_percent'].rolling(window=25).sum()

    # Time features
    df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S", errors='coerce')
    df['hour_fraction'] = df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600
    df['sin_time'] = np.sin(2 * np.pi * df['hour_fraction'] / 24.0)
    df['cos_time'] = np.cos(2 * np.pi * df['hour_fraction'] / 24.0)

    # Fill missing
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    return df


# ==== FUNCTION: Train and Save Model ====
def train_and_save_model(X, y):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # XGBoost classifier
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))
    print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("🔍 ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    # Threshold tuning
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"🎯 Best Threshold = {best_threshold:.4f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

    # Save threshold tuning plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', marker='o')
    plt.plot(thresholds, precision[:-1], label='Precision', linestyle='--', marker='x')
    plt.plot(thresholds, recall[:-1], label='Recall', linestyle='--', marker='s')
    plt.axvline(x=best_threshold, color='gray', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Threshold Tuning on Classifier Output')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(THRESHOLD_PLOT_PATH)
    plt.close()
    print(f"📈 Threshold tuning plot saved to: {THRESHOLD_PLOT_PATH}")


# ==== MAIN ====
def main():
    print("🚀 Loading and preprocessing data...")
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
        print("❌ ERROR: Target has only one class. Exiting.")
        return

    train_and_save_model(X, y)


if __name__ == "__main__":
    main()
