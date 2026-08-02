"""
XGBoost Training for Mars Vortex Detection
==========================================

This script trains an XGBoost (eXtreme Gradient Boosting) classifier for
on-board vortex detection. It mirrors the Random Forest pipeline in
"core pipeline scripts/train_rf_model.py" (same datasets, same features,
same evaluation reports) so results are directly comparable, while taking
advantage of XGBoost-specific capabilities:

- Sequential boosting: each tree corrects the errors of the previous ones
- scale_pos_weight for class imbalance (XGBoost's equivalent of
  class_weight='balanced')
- Early stopping on the validation set to pick the optimal number of trees
- Built-in L1/L2 regularization (alpha/lambda) and gamma pruning
- Histogram-based tree construction ('hist') for fast training
- Native handling of missing values (no imputation needed)
- Gain-based feature importance (more informative than split counts)
- Portable native model format (.json) suitable for on-board deployment

Designed for deployment on Qualcomm Snapdragon-class processors.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
    precision_recall_curve
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Resolve paths relative to the project root (parent of this folder) so the
# script works no matter where it is launched from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")


class Config:
    """Training configuration."""

    # File paths (same datasets as the Random Forest pipeline)
    TRAIN_FILE = os.path.join(DATASETS_DIR, "train_features.csv")
    VAL_FILE = os.path.join(DATASETS_DIR, "val_features.csv")
    TEST_FILE = os.path.join(DATASETS_DIR, "test_features.csv")

    # Output directories (kept inside the xgboost folder)
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "models")
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

    # XGBoost parameters
    #
    # n_estimators is set high on purpose: early stopping on the validation
    # set decides the actual number of boosting rounds, so the model never
    # trains longer than it helps.
    #
    # scale_pos_weight is computed at runtime from the training labels
    # (n_negative / n_positive) — see compute_scale_pos_weight().
    XGB_PARAMS = {
        'n_estimators': 1000,          # upper bound; early stopping picks the real one
        'max_depth': 6,                # boosted trees stay shallow (RF used 15)
        'learning_rate': 0.05,         # low eta + many trees = robust ensemble
        'subsample': 0.8,              # row sampling per tree (fights overfitting)
        'colsample_bytree': 0.8,       # feature sampling per tree
        'min_child_weight': 5,         # counterpart of RF's min_samples_leaf
        'gamma': 0.1,                  # minimum loss reduction to split (pruning)
        'reg_lambda': 1.0,             # L2 regularization on leaf weights
        'reg_alpha': 0.0,              # L1 regularization (sparsity if > 0)
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',        # PR-AUC suits the imbalanced vortex problem
        'tree_method': 'hist',         # fast histogram-based construction
        'random_state': 42,
        'n_jobs': -1
    }

    # Early stopping: stop if validation PR-AUC does not improve for N rounds
    EARLY_STOPPING_ROUNDS = 50

    # Feature columns (exclude metadata) — identical to the RF pipeline
    EXCLUDE_COLUMNS = ['window_id', 'event_sclk', 'label']

    # Random seed
    RANDOM_SEED = 42

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load training, validation, and test datasets."""
    print("Loading datasets...")

    train_df = pd.read_csv(Config.TRAIN_FILE)
    val_df = pd.read_csv(Config.VAL_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)

    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    return train_df, val_df, test_df


def prepare_features(df):
    """
    Prepare features and labels from DataFrame.

    Args:
        df: DataFrame with features and label

    Returns:
        X: Feature matrix
        y: Label vector
        feature_cols: List of feature column names
    """
    feature_cols = [col for col in df.columns if col not in Config.EXCLUDE_COLUMNS]

    X = df[feature_cols].values
    y = df['label'].values

    return X, y, feature_cols


def compute_scale_pos_weight(y_train):
    """
    Compute scale_pos_weight = n_negative / n_positive.

    This is XGBoost's mechanism for class imbalance: gradients of the
    positive class are multiplied by this factor, playing the same role
    as class_weight='balanced' in the Random Forest.
    """
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))

    if n_pos == 0:
        raise ValueError("Training set contains no positive samples.")

    spw = n_neg / n_pos
    print(f"  Class balance: {n_neg} negative / {n_pos} positive "
          f"-> scale_pos_weight = {spw:.3f}")
    return spw

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val, param_overrides=None):
    """
    Train XGBoost classifier with early stopping on the validation set.

    Unlike the Random Forest (independent trees), XGBoost builds trees
    sequentially: each new tree is fit on the errors (gradients) of the
    ensemble so far. The validation set is watched at every round and
    training stops once it stops improving.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (used ONLY for early stopping;
                      final threshold-independent metrics on it remain
                      comparable to the RF pipeline's usage)
        param_overrides: Optional dict of hyperparameters that replace
                         the Config defaults (e.g. the winner of the
                         tune_xgb_hyperparams.py search)

    Returns:
        Trained XGBClassifier, training time in seconds
    """
    print("\nTraining XGBoost classifier...")

    params = dict(Config.XGB_PARAMS)
    if param_overrides:
        print(f"  Applying tuned overrides: {param_overrides}")
        params.update(param_overrides)
    params['scale_pos_weight'] = compute_scale_pos_weight(y_train)

    print(f"  Parameters: {params}")

    start_time = time.time()

    # xgboost >= 1.6 takes early_stopping_rounds in the constructor;
    # older versions (>= 1.5, per requirements.txt) take it in fit().
    try:
        model = XGBClassifier(**params,
                              early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    except TypeError:
        model = XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                  verbose=False)

    training_time = time.time() - start_time

    best_iteration = getattr(model, 'best_iteration', None)
    best_score = getattr(model, 'best_score', None)

    print(f"  Training completed in {training_time:.2f} seconds")
    if best_iteration is not None:
        print(f"  Early stopping: best iteration = {best_iteration} "
              f"(out of max {params['n_estimators']})")
    if best_score is not None:
        print(f"  Best validation {params['eval_metric']}: {best_score:.4f}")

    return model, training_time

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X, y, split_name="Validation", threshold=0.5):
    """
    Evaluate model performance with comprehensive metrics.

    Identical report format to the Random Forest pipeline so the two
    models can be compared line by line.

    Args:
        model: Trained model
        X: Features
        y: True labels
        split_name: Name of the split (for reporting)
        threshold: Decision threshold applied to P(vortex). 0.5 is the
                   library default, but with scale_pos_weight the output
                   probabilities are skewed on purpose, so the tuned
                   threshold from tune_decision_threshold() is what
                   should be used for real decisions.

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {split_name.upper()} SET (threshold = {threshold:.4f})")
    print(f"{'='*70}")

    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"True class distribution: {class_dist}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"True Negative   {cm[0,0]:<20} {cm[0,1]:<20}")
    print(f"True Positive   {cm[1,0]:<20} {cm[1,1]:<20}")

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))

    # Detailed Metrics
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary')

    print(f"\nDetailed Metrics (Positive Class):")
    print(f"  Precision: {precision:.4f} (of predicted positives, how many are correct)")
    print(f"  Recall:    {recall:.4f} (of true positives, how many were detected)")
    print(f"  F1-Score:  {f1:.4f} (harmonic mean of precision and recall)")

    # ROC AUC
    try:
        auc = roc_auc_score(y, y_pred_proba)
        print(f"  ROC AUC:   {auc:.4f}")
    except Exception:
        auc = None
        print(f"  ROC AUC:   N/A (insufficient data)")

    # False Positive/Negative Analysis
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]
    true_negatives = cm[0, 0]

    print(f"\nError Analysis:")
    print(f"  False Positives: {false_positives} (incorrectly predicted as vortex)")
    print(f"  False Negatives: {false_negatives} (missed vortex detections)")
    print(f"  True Positives:  {true_positives} (correctly detected vortices)")
    print(f"  True Negatives:  {true_negatives} (correctly identified non-vortex)")

    metrics = {
        'split': split_name,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

    return metrics

# =============================================================================
# DECISION THRESHOLD TUNING
# =============================================================================

def tune_decision_threshold(model, X_val, y_val):
    """
    Find the decision threshold that maximizes F1 on the validation set.

    Why this matters: model.predict() cuts P(vortex) at 0.5, but
    scale_pos_weight deliberately inflates positive-class probabilities
    to handle imbalance, so 0.5 is arbitrary. The project's Random
    Forest work found its F1-optimal threshold near 0.90 — the same
    calibration step is applied here for XGBoost.

    The full precision/recall/F1 sweep is returned so it can be saved
    and inspected (results/xgb_threshold_sweep.csv), and alternative
    operating points (high-precision, high-recall) are reported for
    mission planning.

    Args:
        model: Trained model
        X_val, y_val: Validation data (threshold must NEVER be tuned
                      on the test set)

    Returns:
        best_threshold (float), sweep_df (DataFrame)
    """
    print(f"\n{'='*70}")
    print("DECISION THRESHOLD TUNING (on validation set)")
    print(f"{'='*70}")

    y_proba = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)

    # precision_recall_curve returns len(thresholds) + 1 precision/recall
    # points; drop the last point (threshold = +inf) to align arrays.
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0.0
        )

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    sweep_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precisions,
        'recall': recalls,
        'f1': f1_scores
    })

    print(f"\n  F1-optimal threshold: {best_threshold:.4f}")
    print(f"    Precision: {precisions[best_idx]:.4f}")
    print(f"    Recall:    {recalls[best_idx]:.4f}")
    print(f"    F1-Score:  {f1_scores[best_idx]:.4f}")
    print(f"  (library default 0.5 is reported for comparison only)")

    # Alternative operating points for mission planning
    hp_mask = precisions >= 0.95
    if hp_mask.any():
        hp_idx = int(np.argmax(np.where(hp_mask, recalls, -1)))
        print(f"\n  High-precision option (P >= 0.95): threshold = "
              f"{thresholds[hp_idx]:.4f} -> P={precisions[hp_idx]:.4f}, "
              f"R={recalls[hp_idx]:.4f}")

    hr_mask = recalls >= 0.90
    if hr_mask.any():
        hr_idx = int(np.argmax(np.where(hr_mask, precisions, -1)))
        print(f"  High-recall option (R >= 0.90):    threshold = "
              f"{thresholds[hr_idx]:.4f} -> P={precisions[hr_idx]:.4f}, "
              f"R={recalls[hr_idx]:.4f}")

    return best_threshold, sweep_df

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model, feature_names):
    """
    Analyze and report feature importance.

    XGBoost offers several importance types; 'gain' (average loss
    reduction brought by splits on the feature) is reported here because
    it is more informative than raw split counts and comparable to the
    impurity-based importance used by the Random Forest.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names

    Returns:
        DataFrame with feature importances (gain and weight)
    """
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE ANALYSIS (gain)")
    print(f"{'='*70}")

    booster = model.get_booster()
    gain_map = booster.get_score(importance_type='gain')
    weight_map = booster.get_score(importance_type='weight')

    # Booster names features f0, f1, ... — map back to real names
    rows = []
    for i, name in enumerate(feature_names):
        key = f'f{i}'
        rows.append({
            'feature': name,
            'gain': gain_map.get(key, 0.0),
            'weight': weight_map.get(key, 0.0)
        })

    importance_df = pd.DataFrame(rows).sort_values('gain', ascending=False)
    importance_df = importance_df.reset_index(drop=True)

    # Normalized gain for easy comparison with RF importances
    total_gain = importance_df['gain'].sum()
    if total_gain > 0:
        importance_df['gain_normalized'] = importance_df['gain'] / total_gain
    else:
        importance_df['gain_normalized'] = 0.0

    print("\nTop 10 Most Important Features (by gain):")
    for i in range(min(10, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"  {i+1}. {row['feature']:25s}: {row['gain_normalized']:.4f} "
              f"(splits: {int(row['weight'])})")

    return importance_df

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, feature_names, training_time, metrics,
               decision_threshold=0.5):
    """
    Save trained model and metadata.

    Two formats are saved:
    - Native XGBoost .json: portable, version-stable, loadable from C/C++
      runtimes — the right format for on-board deployment.
    - joblib .pkl: drop-in parity with the Random Forest pipeline for
      Python-side evaluation scripts.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        training_time: Training duration in seconds
        metrics: Dictionary of validation metrics
        decision_threshold: Tuned decision threshold (saved with the
            model so deployment code applies the SAME cutoff — a model
            without its threshold is only half a detector)
    """
    print(f"\n{'='*70}")
    print("SAVING MODEL")
    print(f"{'='*70}")

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Native XGBoost format (deployment)
    native_path = os.path.join(Config.OUTPUT_DIR, f"xgb_vortex_detector_{timestamp}.json")
    model.save_model(native_path)
    print(f"  Native model saved to: {native_path}")

    # joblib format (parity with RF pipeline)
    pkl_path = os.path.join(Config.OUTPUT_DIR, f"xgb_vortex_detector_{timestamp}.pkl")
    joblib.dump(model, pkl_path)
    print(f"  Pickle model saved to: {pkl_path}")

    # Metadata
    spw = model.get_params().get('scale_pos_weight')
    metadata = {
        'timestamp': timestamp,
        'model_type': 'XGBClassifier',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'xgb_params': model.get_params(),
        'scale_pos_weight': float(spw) if spw is not None else None,
        'early_stopping_rounds': Config.EARLY_STOPPING_ROUNDS,
        'best_iteration': getattr(model, 'best_iteration', None),
        'decision_threshold': decision_threshold,
        'training_time_seconds': training_time,
        'validation_metrics': {
            'threshold_used': metrics.get('threshold', 0.5),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'roc_auc': metrics['auc']
        }
    }

    metadata_path = os.path.join(Config.OUTPUT_DIR, f"model_metadata_{timestamp}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to: {metadata_path}")

    return native_path

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main(param_overrides=None):
    """
    Main training pipeline.

    Args:
        param_overrides: Optional dict of hyperparameters overriding the
            Config defaults (used by run.py --tune to apply the search
            winner automatically).

    Returns:
        Dict with the trained model, tuned threshold, and feature names —
        so run.py can chain the sliding-window evaluation without
        reloading anything from disk.
    """
    print("="*70)
    print("XGBOOST TRAINING - MARS VORTEX DETECTION")
    print("="*70)

    np.random.seed(Config.RANDOM_SEED)

    # Load data
    train_df, val_df, test_df = load_data()

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    X_test, y_test, _ = prepare_features(test_df)

    print(f"  Feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}")

    # Train model (validation set drives early stopping)
    model, training_time = train_xgboost(X_train, y_train, X_val, y_val,
                                         param_overrides=param_overrides)

    # Tune the decision threshold on the validation set
    best_threshold, sweep_df = tune_decision_threshold(model, X_val, y_val)

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    sweep_path = os.path.join(Config.RESULTS_DIR, "xgb_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Threshold sweep saved to: {sweep_path}")

    # Evaluate on validation set with the tuned threshold.
    # NOTE: validation drove early stopping AND threshold selection, so
    # these numbers are optimistic — the test set below is the only
    # untouched measurement.
    val_metrics = evaluate_model(model, X_val, y_val, "Validation",
                                 threshold=best_threshold)

    # Final evaluation on the untouched test set: tuned threshold is the
    # headline number; default 0.5 is reported for comparison.
    test_metrics = evaluate_model(model, X_test, y_test, "Test",
                                  threshold=best_threshold)
    test_metrics_default = evaluate_model(model, X_test, y_test,
                                          "Test (default 0.5)", threshold=0.5)

    # Feature importance
    importance_df = analyze_feature_importance(model, feature_names)

    importance_path = os.path.join(Config.RESULTS_DIR, "xgb_feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")

    # Save model (threshold travels with it)
    model_path = save_model(model, feature_names, training_time, val_metrics,
                            decision_threshold=best_threshold)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nModel ready for deployment:")
    print(f"  Model file: {model_path}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Tuned decision threshold: {best_threshold:.4f}")
    print(f"  Validation F1-Score: {val_metrics['f1']:.4f} (tuned threshold; optimistic)")
    print(f"  Test F1-Score: {test_metrics['f1']:.4f} (tuned threshold — headline number)")
    print(f"  Test F1-Score: {test_metrics_default['f1']:.4f} (default 0.5, for comparison)")

    return {
        'model': model,
        'model_path': model_path,
        'threshold': best_threshold,
        'feature_names': feature_names,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


if __name__ == "__main__":
    main()
