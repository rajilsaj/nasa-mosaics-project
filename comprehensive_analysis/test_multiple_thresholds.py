#!/usr/bin/env python3
"""
Test Comprehensive Models on Multiple Thresholds (EXPLORATORY ONLY)
===================================================================

⚠️  WARNING: This script is for EXPLORATORY ANALYSIS ONLY!
    It shows performance at different thresholds on test set.
    
    DO NOT use this to find optimal thresholds (data leakage!)
    
    For proper evaluation:
    1. Use tune_threshold_validation.py to find optimal threshold on validation set
    2. Use evaluate_test_with_validation_threshold.py to evaluate test set
    
    This script is kept for:
    - Comparing performance across different threshold values
    - Understanding threshold sensitivity
    - Exploratory analysis (NOT for optimization)
"""

import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Multiple thresholds to test
THRESHOLDS = [0.01, 0.02, 0.03]

# =============================================================================
# LOAD MODELS AND DATA
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
        print(f"\nClass Distribution:")
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
# MAIN
# =============================================================================

def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("TESTING COMPREHENSIVE MODELS ON MULTIPLE THRESHOLDS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n⚠️  WARNING: EXPLORATORY ANALYSIS ONLY!")
    print(f"   This script shows performance at different thresholds on TEST set.")
    print(f"   DO NOT use this to find optimal thresholds (data leakage!)")
    print(f"\n   For proper evaluation:")
    print(f"   1. Run tune_threshold_validation.py (tune on validation)")
    print(f"   2. Run evaluate_test_with_validation_threshold.py (evaluate test)")
    print(f"\n   This script is for comparison/exploration only.")
    print(f"\nThresholds to test: {THRESHOLDS}")
    
    # Load models and data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, feature_cols = load_models_and_data()
    
    if baseline_model is None:
        return 1
    
    print("\n" + "=" * 70)
    print("EVALUATING AT MULTIPLE THRESHOLDS")
    print("=" * 70)
    
    # Evaluate baseline model
    print("\n" + "-" * 70)
    print("BASELINE MODEL (15 features)")
    print("-" * 70)
    
    baseline_results = []
    for threshold in THRESHOLDS:
        result = evaluate_at_threshold(baseline_model, baseline_X, y, threshold)
        baseline_results.append(result)
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Accuracy:  {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
        print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
        print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
        print(f"  ROC AUC:   {result['roc_auc']:.4f}" if result['roc_auc'] else "  ROC AUC:   N/A")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, TN: {result['tn']}")
    
    # Evaluate autoencoder model
    print("\n" + "-" * 70)
    print("AUTOENCODER MODEL (19 features)")
    print("-" * 70)
    
    autoencoder_results = []
    for threshold in THRESHOLDS:
        result = evaluate_at_threshold(autoencoder_model, autoencoder_X, y, threshold)
        autoencoder_results.append(result)
        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Accuracy:  {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        print(f"  Precision: {result['precision']:.4f} ({result['precision']*100:.2f}%)")
        print(f"  Recall:    {result['recall']:.4f} ({result['recall']*100:.2f}%)")
        print(f"  F1-Score:  {result['f1_score']:.4f} ({result['f1_score']*100:.2f}%)")
        print(f"  ROC AUC:   {result['roc_auc']:.4f}" if result['roc_auc'] else "  ROC AUC:   N/A")
        print(f"  TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}, TN: {result['tn']}")
    
    # Summary table - Focused on requested metrics
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - BASELINE MODEL")
    print("=" * 70)
    print(f"{'Threshold':<12} {'ROC AUC':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for result in baseline_results:
        roc_auc_str = f"{result['roc_auc']*100:.2f}%" if result['roc_auc'] else "N/A"
        print(f"{result['threshold']:<12.2f} {roc_auc_str:<12} {result['precision']*100:<12.2f}% "
              f"{result['recall']*100:<12.2f}% {result['f1_score']*100:<12.2f}%")
    
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - AUTOENCODER MODEL")
    print("=" * 70)
    print(f"{'Threshold':<12} {'ROC AUC':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)
    for result in autoencoder_results:
        roc_auc_str = f"{result['roc_auc']*100:.2f}%" if result['roc_auc'] else "N/A"
        print(f"{result['threshold']:<12.2f} {roc_auc_str:<12} {result['precision']*100:<12.2f}% "
              f"{result['recall']*100:<12.2f}% {result['f1_score']*100:<12.2f}%")
    
    # Detailed metrics table
    print("\n" + "=" * 70)
    print("DETAILED METRICS - BASELINE MODEL")
    print("=" * 70)
    for result in baseline_results:
        print(f"\nThreshold: {result['threshold']:.2f}")
        print(f"  ROC AUC:   {result['roc_auc']*100:.2f}%" if result['roc_auc'] else "  ROC AUC:   N/A")
        print(f"  Precision: {result['precision']*100:.2f}%")
        print(f"  Recall:    {result['recall']*100:.2f}%")
        print(f"  F1-Score:  {result['f1_score']*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("DETAILED METRICS - AUTOENCODER MODEL")
    print("=" * 70)
    for result in autoencoder_results:
        print(f"\nThreshold: {result['threshold']:.2f}")
        print(f"  ROC AUC:   {result['roc_auc']*100:.2f}%" if result['roc_auc'] else "  ROC AUC:   N/A")
        print(f"  Precision: {result['precision']*100:.2f}%")
        print(f"  Recall:    {result['recall']*100:.2f}%")
        print(f"  F1-Score:  {result['f1_score']*100:.2f}%")
    
    # Try to load validation-tuned thresholds for comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH VALIDATION-TUNED THRESHOLDS")
    print("=" * 70)
    
    threshold_files = glob.glob(os.path.join(RESULTS_DIR, "threshold_tuning_validation_*.json"))
    val_baseline_thresh = None
    val_autoencoder_thresh = None
    
    if threshold_files:
        latest_file = max(threshold_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            tuning_results = json.load(f)
        
        val_baseline_thresh = tuning_results['baseline_model']['optimal_threshold']
        val_autoencoder_thresh = tuning_results['autoencoder_model']['optimal_threshold']
        
        print(f"\n✅ Validation-tuned thresholds (from {os.path.basename(latest_file)}):")
        print(f"   Baseline:    {val_baseline_thresh:.4f}")
        print(f"   Autoencoder: {val_autoencoder_thresh:.4f}")
        print(f"\n   ⚠️  Use these thresholds for proper evaluation (not the 'best' from test set!)")
    else:
        print("\n⚠️  No validation tuning results found.")
        print("   Run tune_threshold_validation.py first for proper threshold tuning!")
    
    # Show performance at validation-tuned thresholds (if available)
    if val_baseline_thresh is not None and val_autoencoder_thresh is not None:
        print("\n" + "-" * 70)
        print("PERFORMANCE AT VALIDATION-TUNED THRESHOLDS (Proper Evaluation)")
        print("-" * 70)
        
        # Find results at validation thresholds
        baseline_val_result = next((r for r in baseline_results if abs(r['threshold'] - val_baseline_thresh) < 0.001), None)
        autoencoder_val_result = next((r for r in autoencoder_results if abs(r['threshold'] - val_autoencoder_thresh) < 0.001), None)
        
        if baseline_val_result:
            print(f"\nBaseline Model (threshold={val_baseline_thresh:.4f}):")
            print(f"  F1-Score:  {baseline_val_result['f1_score']*100:.2f}%")
            print(f"  Precision: {baseline_val_result['precision']*100:.2f}%")
            print(f"  Recall:    {baseline_val_result['recall']*100:.2f}%")
        
        if autoencoder_val_result:
            print(f"\nAutoencoder Model (threshold={val_autoencoder_thresh:.4f}):")
            print(f"  F1-Score:  {autoencoder_val_result['f1_score']*100:.2f}%")
            print(f"  Precision: {autoencoder_val_result['precision']*100:.2f}%")
            print(f"  Recall:    {autoencoder_val_result['recall']*100:.2f}%")
    
    # REMOVED: Finding "best" threshold on test set (data leakage!)
    # This was causing the problem - we should NOT optimize on test set
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT REMINDER")
    print("=" * 70)
    print("   This script shows performance at different thresholds for EXPLORATION.")
    print("   DO NOT use the 'best' threshold from test set (data leakage!)")
    print("   Use validation-tuned thresholds for proper evaluation.")
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"test_multiple_thresholds_{timestamp}.json")
    
    results = {
        'timestamp': timestamp,
        'thresholds_tested': THRESHOLDS,
        'warning': 'EXPLORATORY ANALYSIS ONLY - Do not use for threshold optimization',
        'baseline_model': {
            'results': baseline_results
        },
        'autoencoder_model': {
            'results': autoencoder_results
        }
    }
    
    # Add validation-tuned thresholds if available
    if val_baseline_thresh is not None and val_autoencoder_thresh is not None:
        results['validation_tuned_thresholds'] = {
            'baseline': float(val_baseline_thresh),
            'autoencoder': float(val_autoencoder_thresh)
        }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("EXPLORATORY ANALYSIS COMPLETED")
    print("=" * 70)
    print("\n📝 Remember:")
    print("   - This was exploratory analysis (showing different thresholds)")
    print("   - For proper evaluation, use validation-tuned thresholds")
    print("   - Run evaluate_test_with_validation_threshold.py for final results")
    
    return 0

if __name__ == "__main__":
    exit(main())

