#!/usr/bin/env python3
"""
Test Models at High Thresholds (0.03+)
======================================

Tests both models at thresholds 0.03 and above to verify
that results are zeros (as expected from probability analysis).
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
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# High thresholds to test
HIGH_THRESHOLDS = [0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.50]

# =============================================================================
# LOAD MODELS AND TEST DATA
# =============================================================================

def load_models_and_data():
    """Load models and test sliding window features."""
    print("=" * 70)
    print("LOADING MODELS AND TEST DATA")
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
    
    # Load test sliding window features
    features_file = os.path.join(FEATURES_DIR, "test_sliding_features_step10.csv")
    if not os.path.exists(features_file):
        print(f"[ERROR] Test features not found: {features_file}")
        return None, None, None, None, None, None
    
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df):,} test feature vectors")
    
    # Check class distribution
    if 'label' in features_df.columns:
        class_dist = features_df['label'].value_counts()
        print(f"\nTest Class Distribution:")
        print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(features_df)*100:.2f}%)")
        print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(features_df)*100:.2f}%)")
    
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
        baseline_features = feature_cols[:15]
    
    if autoencoder_metadata_files:
        with open(max(autoencoder_metadata_files, key=os.path.getctime), 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = feature_cols
    
    # Select features for each model
    baseline_feature_cols = [f for f in baseline_features if f in feature_cols]
    autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
    
    baseline_X = features_df[baseline_feature_cols].values
    autoencoder_X = features_df[autoencoder_feature_cols].values
    y = features_df['label'].values
    
    print(f"\nBaseline features: {len(baseline_feature_cols)}")
    print(f"Autoencoder features: {len(autoencoder_feature_cols)}")
    
    return baseline_model, autoencoder_model, baseline_X, autoencoder_X, y

# =============================================================================
# EVALUATE AT THRESHOLD
# =============================================================================

def evaluate_at_threshold(model, X, y, threshold):
    """Evaluate model at a specific threshold."""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Check max probability
    max_proba = y_proba.max()
    
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
        'max_probability': float(max_proba),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'num_predictions': int(y_pred.sum())
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("TESTING MODELS AT HIGH THRESHOLDS (0.03+)")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTesting thresholds: {HIGH_THRESHOLDS}")
    print(f"Expected: Autoencoder model should produce zeros at 0.03+")
    print(f"         (max probability = 2.88%, which is < 3%)")
    
    # Load models and data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y = load_models_and_data()
    
    if baseline_model is None:
        return 1
    
    # ========================================================================
    # TEST BASELINE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("BASELINE MODEL (15 features) - HIGH THRESHOLDS")
    print("=" * 70)
    
    baseline_results = []
    for threshold in HIGH_THRESHOLDS:
        result = evaluate_at_threshold(baseline_model, baseline_X, y, threshold)
        baseline_results.append(result)
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Max Probability: {result['max_probability']:.6f} ({result['max_probability']*100:.4f}%)")
        print(f"  Predictions: {result['num_predictions']} positive predictions")
        print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
        print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
        print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, TN: {result['tn']}")
        
        if result['num_predictions'] == 0:
            print(f"  ⚠️  ZERO PREDICTIONS - All samples predicted as negative")
    
    # ========================================================================
    # TEST AUTOENCODER MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("AUTOENCODER MODEL (19 features) - HIGH THRESHOLDS")
    print("=" * 70)
    
    autoencoder_results = []
    for threshold in HIGH_THRESHOLDS:
        result = evaluate_at_threshold(autoencoder_model, autoencoder_X, y, threshold)
        autoencoder_results.append(result)
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Max Probability: {result['max_probability']:.6f} ({result['max_probability']*100:.4f}%)")
        print(f"  Predictions: {result['num_predictions']} positive predictions")
        print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
        print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
        print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, TN: {result['tn']}")
        
        if result['num_predictions'] == 0:
            print(f"  ⚠️  ZERO PREDICTIONS - All samples predicted as negative")
    
    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - BASELINE MODEL")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Max Prob':<12} {'Predictions':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 75)
    for result in baseline_results:
        max_prob_str = f"{result['max_probability']*100:.2f}%"
        pred_str = f"{result['num_predictions']}"
        prec_str = f"{result['precision']*100:.2f}%" if result['precision'] > 0 else "0.00%"
        rec_str = f"{result['recall']*100:.2f}%" if result['recall'] > 0 else "0.00%"
        f1_str = f"{result['f1_score']*100:.2f}%" if result['f1_score'] > 0 else "0.00%"
        print(f"{result['threshold']:<12.2f} {max_prob_str:<12} {pred_str:<15} {prec_str:<12} {rec_str:<12} {f1_str:<12}")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - AUTOENCODER MODEL")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Max Prob':<12} {'Predictions':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 75)
    for result in autoencoder_results:
        max_prob_str = f"{result['max_probability']*100:.2f}%"
        pred_str = f"{result['num_predictions']}"
        prec_str = f"{result['precision']*100:.2f}%" if result['precision'] > 0 else "0.00%"
        rec_str = f"{result['recall']*100:.2f}%" if result['recall'] > 0 else "0.00%"
        f1_str = f"{result['f1_score']*100:.2f}%" if result['f1_score'] > 0 else "0.00%"
        print(f"{result['threshold']:<12.2f} {max_prob_str:<12} {pred_str:<15} {prec_str:<12} {rec_str:<12} {f1_str:<12}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"high_threshold_test_{timestamp}.json")
    
    results = {
        'timestamp': timestamp,
        'thresholds_tested': HIGH_THRESHOLDS,
        'baseline_model': {
            'results': baseline_results
        },
        'autoencoder_model': {
            'results': autoencoder_results
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())





