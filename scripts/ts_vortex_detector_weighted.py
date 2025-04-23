# TS Vortex Detector Weighted
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from vortexdetect.visualize_metrics import visualize_model_metrics, create_model_report
from vortexdetect.analyze_overlap import analyze_pressure_patterns


def create_temporal_features(sequence):
    """Create 5 statistical features from a pressure difference window."""
    recent_trend = np.mean(sequence[-20:-10]) if len(sequence) >= 20 else 0
    slope = np.polyfit(range(len(sequence)), sequence, 1)[0]
    return np.array([
        np.mean(sequence),
        np.std(sequence),
        np.sum(sequence < 0),
        recent_trend,
        slope
    ])


def load_and_prepare_data(file_path: Path, window_size=50, negative_ratio=10):
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df['pressure_diff'] = df['PRESSURE'].diff().shift(1)

    vortex_events = df[df['gt_fwhm'] > 0].index
    vortex_groups = []
    current_group = [vortex_events[0]]
    for i in range(1, len(vortex_events)):
        if vortex_events[i] - vortex_events[i - 1] == 1:
            current_group.append(vortex_events[i])
        else:
            vortex_groups.append(current_group)
            current_group = [vortex_events[i]]
    vortex_groups.append(current_group)

    X, y = [], []

    for group in tqdm(vortex_groups, desc="Positive samples"):
        start_idx = group[0]
        detection_windows = df[df['gt_detection_win'] > 0].index
        detection_before = detection_windows[detection_windows < start_idx]
        if len(detection_before) > 0:
            last_detection = detection_before[-1]
            if last_detection >= window_size:
                seq = df['pressure_diff'].iloc[last_detection - window_size:last_detection].dropna().values
                if len(seq) == window_size:
                    X.append(create_temporal_features(seq))
                    y.append(1)

    n_pos = len(y)
    n_neg = n_pos * negative_ratio
    non_detection = df[(df['gt_detection_win'] == 0) & (df.index >= window_size)].index
    sampled = np.random.choice(non_detection, min(n_neg, len(non_detection)), replace=False)

    for idx in tqdm(sampled, desc="Negative samples"):
        seq = df['pressure_diff'].iloc[idx - window_size:idx].dropna().values
        if len(seq) == window_size:
            X.append(create_temporal_features(seq))
            y.append(0)

    return np.array(X), np.array(y), df


def train_and_evaluate(X, y, df):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    weight_scenarios = [
        'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}, {0: 1, 1: 5}, {0: 1, 1: 10}
    ]

    results_dir = Path("results") / "weighted_rf"
    results_dir.mkdir(parents=True, exist_ok=True)

    for weight in weight_scenarios:
        print(f"\n🧪 Testing with class_weight = {weight}")
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weight)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        energy_saved = (1 - np.mean(y_pred)) * 100
        data_quality = precision * 100

        print(f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"Energy Saved: {energy_saved:.2f}% | Data Quality: {data_quality:.2f}%")

        model_name = f"RF_Weights_{str(weight).replace(':', '_')}"
        visualize_model_metrics(
            model=model,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_pred_proba=y_prob,
            model_name=model_name,
            save_dir=results_dir
        )

        create_model_report(model_name, results_dir)

        patterns = analyze_pressure_patterns(df, y_test, y_pred)
        for k, v in patterns.items():
            print(f"\nPattern stats for {k}:")
            for metric, val in v.items():
                print(f"{metric}: {val:.4f}")


def main():
    print("🚀 Starting weighted vortex detection experiment")
    start = time.time()

    data_path = Path("data") / "ml_ready_vortex_data.csv"
    X, y, df = load_and_prepare_data(data_path)
    train_and_evaluate(X, y, df)

    print(f"\n✅ Done in {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
