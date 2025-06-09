import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score,
    accuracy_score, roc_curve
)
from xgboost import XGBClassifier

# === Paths ===
DATA_PATH = "data/ml_ready_vortex_data.csv"
MODEL_PATH = "results/xgboost_vortex_model.pkl"
CONF_MATRIX_PATH = "results/confusion_matrix.png"
THRESHOLD_PLOT_PATH = "results/threshold_tuning.png"
ROC_CURVE_PATH = "results/roc_curve.png"
os.makedirs("results", exist_ok=True)

# === Preprocessing ===
def preprocess(df):
    df['pressure_diff'] = df['PRESSURE'] - df['PRESSURE_MA_500']
    df['pressure_diff_percent'] = (df['pressure_diff'] / df['PRESSURE_MA_500']) * 100

    for window in [10, 25]:
        df[f'rolling_mean_{window}'] = df['pressure_diff'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['pressure_diff'].rolling(window).std()

    for lag in [1, 2]:
        df[f'pressure_lag_{lag}'] = df['PRESSURE'].shift(lag)

    x = np.arange(25)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def fast_slope(series):
        if np.isnan(series).any():
            return np.nan
        return ((x - x_mean) * (series - series.mean())).sum() / denom

    df['slope_25'] = df['PRESSURE'].rolling(window=25).apply(fast_slope, raw=True)
    df['drop_1_percent'] = (df['pressure_diff_percent'] < -1.0).astype(int)
    df['count_drops_25'] = df['drop_1_percent'].rolling(window=25).sum()

    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
    df['hour_fraction'] = df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600
    df['sin_time'] = np.sin(2 * np.pi * df['hour_fraction'] / 24)
    df['cos_time'] = np.cos(2 * np.pi * df['hour_fraction'] / 24)

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    return df

# === Training with K-Fold CV ===
def train_with_kfold(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, precs, recalls, f1s, aucs = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n🌀 Fold {fold}:")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_proba))

        if fold == n_splits:
            joblib.dump(model, MODEL_PATH)
            print(f"✅ Model saved to: {MODEL_PATH}")

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(CONF_MATRIX_PATH)
            plt.close()
            print(f"📊 Confusion matrix saved to: {CONF_MATRIX_PATH}")

            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
            best_threshold = thresholds[np.argmax(f1_scores)]

            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, f1_scores[:-1], label='F1 Score')
            plt.plot(thresholds, precision[:-1], linestyle='--', label='Precision')
            plt.plot(thresholds, recall[:-1], linestyle='--', label='Recall')
            plt.axvline(x=best_threshold, color='red', linestyle='--', label=f'Best Threshold: {best_threshold:.4f}')
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.title("Threshold Tuning")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(THRESHOLD_PLOT_PATH)
            plt.close()
            print(f"📈 Threshold tuning plot saved to: {THRESHOLD_PLOT_PATH}")

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.4f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(ROC_CURVE_PATH)
            plt.close()
            print(f"📈 ROC curve saved to: {ROC_CURVE_PATH}")

    print("\n🔁 Cross-Validation Results (Average of All Folds):")
    print(f" - Accuracy:  {np.mean(accs):.4f}")
    print(f" - Precision: {np.mean(precs):.4f}")
    print(f" - Recall:    {np.mean(recalls):.4f}")
    print(f" - F1 Score:  {np.mean(f1s):.4f}")
    print(f" - ROC-AUC:   {np.mean(aucs):.4f}")

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
        print("❌ ERROR: Only one class in target.")
        return

    train_with_kfold(X, y)

if __name__ == "__main__":
    main()
