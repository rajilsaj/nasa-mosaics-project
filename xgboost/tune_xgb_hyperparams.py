"""
XGBoost Hyperparameter Tuning for Mars Vortex Detection
=======================================================

Searches over the key XGBoost hyperparameters using the project's
temporal train/validation split (NOT k-fold cross-validation — shuffling
time-series windows across folds would leak future data into training,
which the temporal split was specifically built to prevent).

Each candidate is trained on train_features.csv with early stopping on
val_features.csv, then scored on the validation set by F1 (positive
class). The best configuration is printed and saved so it can be copied
into Config.XGB_PARAMS in train_xgb_model.py.

Run AFTER confirming the baseline in train_xgb_model.py works.
"""

import pandas as pd
import numpy as np
import os
import json
import itertools
import time
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

TRAIN_FILE = os.path.join(DATASETS_DIR, "train_features.csv")
VAL_FILE = os.path.join(DATASETS_DIR, "val_features.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

EXCLUDE_COLUMNS = ['window_id', 'event_sclk', 'label']
EARLY_STOPPING_ROUNDS = 50
RANDOM_SEED = 42

# Fixed parameters shared by every candidate
BASE_PARAMS = {
    'n_estimators': 1000,           # early stopping picks the real count
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr',
    'tree_method': 'hist',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Search space — 3 x 3 x 2 x 2 x 2 x 2 = 144 candidates.
# Trim lists to shrink the search if runtime is a concern.
PARAM_GRID = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1],
    'min_child_weight': [1, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.2],
}

# =============================================================================
# DATA
# =============================================================================

def load_split(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    return df[feature_cols].values, df['label'].values, feature_cols


def fit_with_early_stopping(params, X_train, y_train, X_val, y_val):
    """Fit one candidate; handles both old and new xgboost early-stopping APIs."""
    try:
        model = XGBClassifier(**params, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    return model

# =============================================================================
# SEARCH
# =============================================================================

def main():
    print("="*70)
    print("XGBOOST HYPERPARAMETER TUNING - MARS VORTEX DETECTION")
    print("="*70)

    np.random.seed(RANDOM_SEED)

    print("\nLoading datasets...")
    X_train, y_train, feature_cols = load_split(TRAIN_FILE)
    X_val, y_val, _ = load_split(VAL_FILE)
    print(f"  Training: {len(y_train)} samples, Validation: {len(y_val)} samples")
    print(f"  Features: {len(feature_cols)}")

    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    scale_pos_weight = n_neg / n_pos
    print(f"  scale_pos_weight = {scale_pos_weight:.3f}")

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*(PARAM_GRID[k] for k in keys)))
    print(f"\nSearching {len(combos)} candidate configurations "
          f"(early stopping after {EARLY_STOPPING_ROUNDS} stale rounds)...\n")

    results = []
    best = None

    start = time.time()
    for i, combo in enumerate(combos, 1):
        params = dict(BASE_PARAMS)
        params.update(dict(zip(keys, combo)))
        params['scale_pos_weight'] = scale_pos_weight

        model = fit_with_early_stopping(params, X_train, y_train, X_val, y_val)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(y_val, y_proba)
        except Exception:
            auc = float('nan')

        record = dict(zip(keys, combo))
        record.update({
            'best_iteration': getattr(model, 'best_iteration', None),
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
            'val_roc_auc': auc
        })
        results.append(record)

        if best is None or f1 > best['val_f1']:
            best = record
            print(f"  [{i}/{len(combos)}] NEW BEST  F1={f1:.4f}  "
                  f"P={precision:.4f}  R={recall:.4f}  {dict(zip(keys, combo))}")
        elif i % 10 == 0:
            elapsed = time.time() - start
            print(f"  [{i}/{len(combos)}] ... ({elapsed:.0f}s elapsed, "
                  f"best F1 so far {best['val_f1']:.4f})")

    elapsed = time.time() - start
    print(f"\nSearch finished in {elapsed:.1f} seconds.")

    # Save full results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_df = pd.DataFrame(results).sort_values('val_f1', ascending=False)
    results_path = os.path.join(RESULTS_DIR, f"xgb_tuning_results_{timestamp}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"All results saved to: {results_path}")

    best_path = os.path.join(RESULTS_DIR, f"xgb_best_params_{timestamp}.json")
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2, default=str)
    print(f"Best configuration saved to: {best_path}")

    print(f"\n{'='*70}")
    print("BEST CONFIGURATION (by validation F1)")
    print(f"{'='*70}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Only the searched hyperparameters — this dict can be passed
    # straight to train_xgb_model.main(param_overrides=...), which is
    # exactly what run.py --tune does automatically.
    best_params = {k: best[k] for k in PARAM_GRID}
    return best_params


if __name__ == "__main__":
    main()
