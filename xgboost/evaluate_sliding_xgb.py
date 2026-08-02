"""
Sliding-Window (Deployment-Realistic) Evaluation for XGBoost
============================================================

The window-level metrics printed during training are measured on curated
windows with an artificial class balance. On-board, the detector sees a
continuous stream of overlapping sliding windows where vortices are rare
(<1%), and the project's Random Forest work showed performance collapses
in that regime (F1 dropped from ~87% to ~9%). Any honest RF-vs-XGBoost
comparison must therefore be made HERE, not on the training-style windows.

This script scores the model on the precomputed sliding-window feature
files (project root):

    val_sliding_features.csv   (~54k windows, natural imbalance)
    test_sliding_features.csv  (~86k windows, natural imbalance)

and reports, at both the tuned threshold and the 0.5 default:
    precision / recall / F1, confusion matrix, PR-AUC, ROC-AUC,
    and false positives per hour (computed from real SCLK timestamps —
    the metric that matters for mission energy budgets).

Usage:
    - Chained automatically by run.py after training (in-memory model).
    - Standalone: `python evaluate_sliding_xgb.py` loads the most recent
      trained model + threshold from the models/ folder.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support,
    roc_auc_score, average_precision_score
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

VAL_SLIDING_FILE = os.path.join(DATASETS_DIR, "val_sliding_features.csv")
TEST_SLIDING_FILE = os.path.join(DATASETS_DIR, "test_sliding_features.csv")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Metadata columns present in the sliding files but not used as features
META_COLUMNS = ['window_id', 'start_idx', 'end_idx',
                'start_sclk', 'end_sclk', 'label']

# =============================================================================
# MODEL LOADING (standalone mode)
# =============================================================================

def load_latest_model():
    """
    Load the most recently trained model (native .json) together with
    its tuned decision threshold and feature list from the matching
    metadata file.
    """
    from xgboost import XGBClassifier

    candidates = sorted(glob.glob(
        os.path.join(MODELS_DIR, "xgb_vortex_detector_*.json")))
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found in {MODELS_DIR}. "
            "Run 'python run.py' (or train_xgb_model.py) first.")

    model_path = candidates[-1]
    timestamp = os.path.basename(model_path).replace(
        "xgb_vortex_detector_", "").replace(".json", "")

    model = XGBClassifier()
    model.load_model(model_path)

    threshold = 0.5
    feature_names = None
    metadata_path = os.path.join(MODELS_DIR, f"model_metadata_{timestamp}.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        threshold = float(metadata.get('decision_threshold', 0.5))
        feature_names = metadata.get('feature_names')

    print(f"Loaded model: {model_path}")
    print(f"  Tuned decision threshold: {threshold:.4f}")
    return model, threshold, feature_names

# =============================================================================
# DATA
# =============================================================================

def load_sliding_split(path, feature_names):
    """
    Load one sliding-window feature file.

    Returns X (in the model's feature order), y (0/1), and the duration
    of the covered period in hours (from real SCLK timestamps).
    """
    df = pd.read_csv(path)

    # Labels may be parsed as bool or as 'True'/'False' strings — map
    # explicitly (str.astype(bool) would mark EVERY non-empty string True)
    y = df['label'].map(
        {True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0}
    )
    if y.isna().any():
        raise ValueError(f"Unrecognized label values in {path}: "
                         f"{df['label'][y.isna()].unique()[:5]}")
    y = y.astype(int).values

    # Select features BY NAME in the training order — never rely on
    # column position, the sliding files have extra metadata columns.
    if feature_names is None:
        feature_names = [c for c in df.columns if c not in META_COLUMNS]
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    X = df[feature_names].values

    # SCLK counts seconds -> observation period in hours
    duration_hours = (df['end_sclk'].max() - df['start_sclk'].min()) / 3600.0

    return X, y, duration_hours


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_split(model, X, y, duration_hours, split_name, thresholds):
    """
    Score one sliding split at each requested threshold.

    Returns a list of result dicts (one per threshold).
    """
    print(f"\n{'='*70}")
    print(f"SLIDING-WINDOW EVALUATION: {split_name.upper()}")
    print(f"{'='*70}")

    n_pos = int(y.sum())
    print(f"  Windows: {len(y):,}  |  True vortex windows: {n_pos:,} "
          f"({100.0 * n_pos / len(y):.2f}%)  |  Period: {duration_hours:.1f} h")

    y_proba = model.predict_proba(X)[:, 1]

    # Threshold-independent metrics
    try:
        roc_auc = roc_auc_score(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
        print(f"  ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f} "
              f"(PR-AUC baseline = positive rate = {n_pos / len(y):.4f})")
    except Exception:
        roc_auc, pr_auc = None, None

    results = []
    for name, thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='binary', zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fp_per_hour = fp / duration_hours if duration_hours > 0 else float('nan')

        print(f"\n  --- {name} (threshold = {thr:.4f}) ---")
        print(f"  Precision: {precision:.4f}   Recall: {recall:.4f}   "
              f"F1: {f1:.4f}")
        print(f"  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
        print(f"  False positives per hour: {fp_per_hour:.2f}")

        results.append({
            'split': split_name,
            'threshold_name': name,
            'threshold': thr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'fp_per_hour': fp_per_hour,
            'n_windows': len(y),
            'n_positive_windows': n_pos,
            'duration_hours': duration_hours
        })

    return results


def run_evaluation(model, threshold, feature_names=None):
    """
    Evaluate a trained model on both sliding splits and save the results.

    Args:
        model: Trained XGBClassifier
        threshold: Tuned decision threshold (from training)
        feature_names: Feature order used at training time

    Returns:
        DataFrame with all results (also saved to
        results/xgb_sliding_window_results.csv)
    """
    print("\n" + "="*70)
    print("DEPLOYMENT-REALISTIC EVALUATION (SLIDING WINDOWS)")
    print("="*70)
    print("Window-level training metrics are optimistic; the numbers below")
    print("reflect the continuous, heavily imbalanced stream the detector")
    print("would actually see on-board.")

    thresholds = [("tuned", threshold), ("default", 0.5)]

    all_results = []
    for split_name, path in [("validation", VAL_SLIDING_FILE),
                             ("test", TEST_SLIDING_FILE)]:
        if not os.path.isfile(path):
            print(f"\n[SKIPPED] {split_name}: {path} not found.")
            continue
        X, y, duration_hours = load_sliding_split(path, feature_names)
        all_results.extend(
            evaluate_split(model, X, y, duration_hours, split_name, thresholds))

    if not all_results:
        print("\nNo sliding-window files found — nothing evaluated.")
        return None

    results_df = pd.DataFrame(all_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "xgb_sliding_window_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSliding-window results saved to: {out_path}")

    # Compact summary
    print(f"\n{'='*70}")
    print("SLIDING-WINDOW SUMMARY (tuned threshold)")
    print(f"{'='*70}")
    tuned = results_df[results_df['threshold_name'] == 'tuned']
    for _, row in tuned.iterrows():
        print(f"  {row['split']:<12} P={row['precision']:.4f}  "
              f"R={row['recall']:.4f}  F1={row['f1']:.4f}  "
              f"FP/hour={row['fp_per_hour']:.2f}")
    print("\nCompare against the RF baseline in the project root")
    print("(threshold_calibration_results.csv: F1~0.12, ~5.5 FP/hour).")

    return results_df


def main():
    """Standalone entry: load the latest saved model and evaluate."""
    model, threshold, feature_names = load_latest_model()
    run_evaluation(model, threshold, feature_names)


if __name__ == "__main__":
    main()
