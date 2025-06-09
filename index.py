"""
Train XGBoost Classifier with Balanced Features and Safe Cleaning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    print("✅ Loaded data:", df.shape)
    return df


def preprocess_data(df):
    # Base features
    df['pressure_diff'] = df['PRESSURE'] - df['PRESSURE_MA_500']
    df['pressure_diff_percent'] = (df['pressure_diff'] / df['PRESSURE_MA_500']) * 100

    # Rolling stats (non-centered, smaller windows)
    for window in [10, 25]:
        df[f'rolling_mean_{window}'] = df['pressure_diff'].rolling(window=window, center=False).mean()
        df[f'rolling_std_{window}'] = df['pressure_diff'].rolling(window=window, center=False).std()

    # Lag features
    for lag in [1, 2]:
        df[f'pressure_lag_{lag}'] = df['PRESSURE'].shift(lag)

    # Optimized slope
    window = 25
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def fast_slope(series):
        if np.isnan(series).any():
            return np.nan
        return ((x - x_mean) * (series - series.mean())).sum() / denom

    df['slope_25'] = df['PRESSURE'].rolling(window=window).apply(fast_slope, raw=True)

    # Drop detection
    df['drop_1_percent'] = (df['pressure_diff_percent'] < -1.0).astype(int)
    df['count_drops_25'] = df['drop_1_percent'].rolling(window=25).sum()

    # Time features
    try:
        df['time'] = pd.to_datetime(df['time'], format="%H:%M:%S", errors='coerce')
        df['hour_fraction'] = df['time'].dt.hour + df['time'].dt.minute / 60 + df['time'].dt.second / 3600
        df['sin_time'] = np.sin(2 * np.pi * df['hour_fraction'] / 24.0)
        df['cos_time'] = np.cos(2 * np.pi * df['hour_fraction'] / 24.0)
    except Exception as e:
        print("⚠️ Time column error:", e)

    print("Class distribution BEFORE fillna:")
    print(df['gt_4xfwhm'].value_counts(dropna=False))

    # Fill missing values (safer than dropna)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)

    print("Class distribution AFTER cleaning:")
    print(df['gt_4xfwhm'].value_counts(dropna=False))

    return df


def build_and_train_model(df):
    features = [
        'PRESSURE', 'PRESSURE_MA_500', 'pressure_diff', 'pressure_diff_percent',
        'rolling_mean_10', 'rolling_std_10', 'rolling_mean_25', 'rolling_std_25',
        'pressure_lag_1', 'pressure_lag_2', 'slope_25',
        'count_drops_25', 'sin_time', 'cos_time'
    ]
    target = 'gt_4xfwhm'

    X = df[features]
    y = df[target]

    if y.nunique() < 2:
        raise ValueError("❌ ERROR: Only one class found in target after preprocessing.")

    # Balance classes
    rus = RandomUnderSampler(random_state=42)
    X_bal, y_bal = rus.fit_resample(X, y)
    print("✅ Balanced dataset:", X_bal.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("🔍 ROC-AUC Score:", roc_auc_score(y_test, y_proba))


def main():
    df = load_data("data/ml_ready_vortex_data.csv")
    df = preprocess_data(df)

    if df['gt_4xfwhm'].nunique() < 2:
        print("❌ Only one class remains. Aborting training.")
        return

    build_and_train_model(df)


if __name__ == "__main__":
    main()
