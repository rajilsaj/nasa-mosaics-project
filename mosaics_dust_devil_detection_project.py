"""
MOSAICS Project: Dust Devil Detection Using Atmospheric Pressure Data
This script performs:
1. Data Loading and Preprocessing
2. Feature Engineering
3. Random Forest and XGBoost Modeling
4. Evaluation with Metrics and Confusion Matrix
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Adjustable Threshold for Vortex Detection
VORTEX_THRESHOLD = -0.3  # Default: -0.3% drop, can be tuned

# 1. Load Data
def load_data(filepath):
    return pd.read_csv(filepath)

# 2. Preprocessing
def preprocess_data(df):
    df.dropna(inplace=True)
    df['pressure_diff'] = df['PRESSURE'] - df['PRESSURE_MA_500']
    df['pressure_diff_percent'] = (df['pressure_diff'] / df['PRESSURE_MA_500']) * 100
    return df

# 3. Feature Engineering
def extract_features(df, window=50):
    df_feat = pd.DataFrame(index=df.index)
    df_feat['mean_pressure'] = df['PRESSURE'].rolling(window=window).mean()
    df_feat['std_pressure'] = df['PRESSURE'].rolling(window=window).std()
    df_feat['max_drop'] = df['pressure_diff'].rolling(window=window).min()
    df_feat['trend'] = df['PRESSURE'].diff().rolling(window=window).mean()

    slopes = []
    for i in range(len(df) - window + 1):
        y = df['PRESSURE'].iloc[i:i+window].values
        x = np.arange(window)
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    df_feat['slope'] = [np.nan] * (window - 1) + slopes

    df_feat.dropna(inplace=True)
    return df_feat

# 4. Label Creation
def generate_labels(df, threshold):
    return (df['pressure_diff_percent'] <= threshold).astype(int)

# 5. Data Split and SMOTE
def balance_and_split(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# 6. Train Random Forest Model
def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

# 7. Train XGBoost Model
def train_xgb(X_train, y_train):
    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# 8. Evaluation Function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filepath = "data/ml_ready_vortex_data.csv"  # Adjust this path if needed
    df = load_data(filepath)
    df = preprocess_data(df)

    print("Minimum pressure_diff_percent in data:", df['pressure_diff_percent'].min())
    print("Total vortex candidates at threshold", VORTEX_THRESHOLD, ":", (df['pressure_diff_percent'] <= VORTEX_THRESHOLD).sum())

    features = extract_features(df)
    features.reset_index(drop=True, inplace=True)

    # Align and regenerate labels
    df = df.iloc[-len(features):].reset_index(drop=True)
    y = generate_labels(df, threshold=VORTEX_THRESHOLD)
    print("Class distribution:", y.value_counts())

    if len(y.unique()) < 2:
        raise ValueError(f"Cannot train model: only one class present in y: {y.unique()}")

    X = features.select_dtypes(include=[np.number])

    X_train, X_test, y_train, y_test = balance_and_split(X, y)

    print("Training Random Forest...")
    rf_model = train_rf(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

    print("Training XGBoost...")
    xgb_model = train_xgb(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test)
