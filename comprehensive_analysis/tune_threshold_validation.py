#!/usr/bin/env python3
"""
Proper Threshold Tuning: Validation Set Only
============================================

This script follows ML best practices:
1. Tune threshold on VALIDATION set only
2. Save optimal threshold
3. Apply to TEST set for final evaluation
4. Never tune on test set (data leakage prevention)
"""

import os
import pandas as pd
import numpy as np
import glob
import json
import joblib
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Thresholds to test on validation set
THRESHOLDS_TO_TEST = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.10]

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

def load_models_and_validation_data():
    """Load models and validation sliding window features."""
    print("=" * 70)
    print("LOADING MODELS AND VALIDATION DATA")
    print("=" * 70)
    
    # Load models
    baseline_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_model_*.pkl"))
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    
    if not baseline_files or not autoencoder_files:
        print("[ERROR] Models not found!")
        return None, None, None, None, None, None
    
    baseline_model = joblib.load(max(baseline_files, key=os.path.getctime))
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    
    print(f"Loaded baseline model: {os.path.basename(max(baseline_files, key=os.path.getctime))}")
    print(f"Loaded autoencoder model: {os.path.basename(max(autoencoder_files, key=os.path.getctime))}")
    
    # Load VALIDATION sliding window features (NOT test!)
    features_file = os.path.join(FEATURES_DIR, "val_sliding_features_step10.csv")
    if not os.path.exists(features_file):
        print(f"[ERROR] Validation features not found: {features_file}")
        return None, None, None, None, None, None
    
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df):,} VALIDATION feature vectors")
    
    # Check class distribution
    if 'label' in features_df.columns:
        class_dist = features_df['label'].value_counts()
        print(f"\nValidation Class Distribution:")
        print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(features_df)*100:.2f}%)")
        print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(features_df)*100:.2f}%)")
        if class_dist.get(1, 0) > 0:
            ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
            print(f"  Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    # Separate features and labels
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    # Load model metadata to get correct feature columns
    baseline_metadata_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_metadata_*.json"))
    autoencoder_metadata_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_metadata_*.json"))
    
    if baseline_metadata_files:
        with open(max(baseline_metadata_files, key=os.path.getctime), 'r') as f:
            baseline_metadata = json.load(f)
            baseline_features = baseline_metadata.get('features', [])
    else:
        baseline_features = feature_cols[:15]  # First 15 features
    
    if autoencoder_metadata_files:
        with open(max(autoencoder_metadata_files, key=os.path.getctime), 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = feature_cols  # All features
    
    # Select features for each model
    baseline_feature_cols = [f for f in baseline_features if f in feature_cols]
    autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
    
    baseline_X = features_df[baseline_feature_cols].values
    autoencoder_X = features_df[autoencoder_feature_cols].values
    y = features_df['label'].values
    
    print(f"\nBaseline features: {len(baseline_feature_cols)}")
    print(f"Autoencoder features: {len(autoencoder_feature_cols)}")
    
    return baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, feature_cols

# =============================================================================
# EVALUATE AT THRESHOLD
# =============================================================================

def evaluate_at_threshold(model, X, y, threshold):
    """Evaluate model at a specific threshold."""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Make predictions at threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    accuracy = (y_pred == y).mean()
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case
        if len(np.unique(y_pred)) == 1:
            if y_pred[0] == 0:
                tn, fp, fn, tp = len(y[y == 0]), 0, len(y[y == 1]), 0
            else:
                tn, fp, fn, tp = 0, len(y[y == 0]), 0, len(y[y == 1])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y, y_proba)
    except ValueError:
        roc_auc = np.nan
    
    return {
        'threshold': threshold,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

# =============================================================================
# FIND OPTIMAL THRESHOLD
# =============================================================================

def find_optimal_threshold(results, metric='f1_score'):
    """Find optimal threshold based on specified metric."""
    if not results:
        return None, None
    
    results_df = pd.DataFrame(results)
    
    # Find threshold that maximizes the metric
    if metric == 'f1_score':
        optimal_idx = results_df['f1_score'].idxmax()
    elif metric == 'precision':
        optimal_idx = results_df['precision'].idxmax()
    elif metric == 'recall':
        optimal_idx = results_df['recall'].idxmax()
    else:
        optimal_idx = results_df['f1_score'].idxmax()
    
    optimal_result = results_df.loc[optimal_idx]
    
    return float(optimal_result['threshold']), optimal_result.to_dict()

# =============================================================================
# MAIN: TUNE ON VALIDATION
# =============================================================================

def main():
    """Main threshold tuning pipeline on validation set."""
    print("=" * 70)
    print("THRESHOLD TUNING ON VALIDATION SET (BEST PRACTICE)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n⚠️  IMPORTANT: Tuning on VALIDATION set only!")
    print(f"   Test set will be evaluated with chosen threshold (no tuning on test)")
    print(f"   Thresholds to test: {THRESHOLDS_TO_TEST}")
    
    # Load models and validation data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y_val, feature_cols = load_models_and_validation_data()
    
    if baseline_model is None:
        return 1
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # TUNE BASELINE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUNING BASELINE MODEL (15 features)")
    print("=" * 70)
    
    baseline_results = []
    for threshold in THRESHOLDS_TO_TEST:
        result = evaluate_at_threshold(baseline_model, baseline_X, y_val, threshold)
        baseline_results.append(result)
        print(f"\nThreshold: {threshold:.3f}")
        print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
        print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
        print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, TN: {result['tn']}")
    
    # Find optimal threshold
    baseline_optimal_threshold, baseline_optimal_metrics = find_optimal_threshold(
        baseline_results, metric='f1_score'
    )
    
    print("\n" + "-" * 70)
    print(f"✅ OPTIMAL THRESHOLD (Baseline): {baseline_optimal_threshold:.4f}")
    print("-" * 70)
    print(f"  Precision: {baseline_optimal_metrics['precision']:.4f} ({baseline_optimal_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {baseline_optimal_metrics['recall']:.4f} ({baseline_optimal_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {baseline_optimal_metrics['f1_score']:.4f} ({baseline_optimal_metrics['f1_score']*100:.2f}%)")
    print(f"  ROC AUC:   {baseline_optimal_metrics['roc_auc']:.4f}" if baseline_optimal_metrics['roc_auc'] else "  ROC AUC:   N/A")
    
    # ========================================================================
    # TUNE AUTOENCODER MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("TUNING AUTOENCODER MODEL (19 features)")
    print("=" * 70)
    
    autoencoder_results = []
    for threshold in THRESHOLDS_TO_TEST:
        result = evaluate_at_threshold(autoencoder_model, autoencoder_X, y_val, threshold)
        autoencoder_results.append(result)
        print(f"\nThreshold: {threshold:.3f}")
        print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
        print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
        print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, TN: {result['tn']}")
    
    # Find optimal threshold
    autoencoder_optimal_threshold, autoencoder_optimal_metrics = find_optimal_threshold(
        autoencoder_results, metric='f1_score'
    )
    
    print("\n" + "-" * 70)
    print(f"✅ OPTIMAL THRESHOLD (Autoencoder): {autoencoder_optimal_threshold:.4f}")
    print("-" * 70)
    print(f"  Precision: {autoencoder_optimal_metrics['precision']:.4f} ({autoencoder_optimal_metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {autoencoder_optimal_metrics['recall']:.4f} ({autoencoder_optimal_metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {autoencoder_optimal_metrics['f1_score']:.4f} ({autoencoder_optimal_metrics['f1_score']*100:.2f}%)")
    print(f"  ROC AUC:   {autoencoder_optimal_metrics['roc_auc']:.4f}" if autoencoder_optimal_metrics['roc_auc'] else "  ROC AUC:   N/A")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING THRESHOLD TUNING RESULTS")
    print("=" * 70)
    
    results_summary = {
        'timestamp': timestamp,
        'tuning_set': 'validation',
        'thresholds_tested': THRESHOLDS_TO_TEST,
        'baseline_model': {
            'optimal_threshold': float(baseline_optimal_threshold),
            'optimal_metrics': baseline_optimal_metrics,
            'all_results': baseline_results
        },
        'autoencoder_model': {
            'optimal_threshold': float(autoencoder_optimal_threshold),
            'optimal_metrics': autoencoder_optimal_metrics,
            'all_results': autoencoder_results
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, f"threshold_tuning_validation_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Saved results to: {results_file}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"  Baseline Model:")
    print(f"    Optimal Threshold: {baseline_optimal_threshold:.4f}")
    print(f"    F1-Score: {baseline_optimal_metrics['f1_score']:.4f}")
    print(f"\n  Autoencoder Model:")
    print(f"    Optimal Threshold: {autoencoder_optimal_threshold:.4f}")
    print(f"    F1-Score: {autoencoder_optimal_metrics['f1_score']:.4f}")
    
    print("\n📝 Next Steps:")
    print(f"  1. Use threshold {baseline_optimal_threshold:.4f} for baseline model on test set")
    print(f"  2. Use threshold {autoencoder_optimal_threshold:.4f} for autoencoder model on test set")
    print(f"  3. Run evaluate_test_with_validation_threshold.py to evaluate test set")
    print(f"  4. DO NOT re-tune on test set (data leakage!)")
    
    return 0

if __name__ == "__main__":
    exit(main())





