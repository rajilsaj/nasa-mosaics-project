#!/usr/bin/env python3
"""
Evaluate Test Set with Validation-Tuned Threshold
=================================================

This script follows ML best practices:
1. Loads optimal threshold from validation tuning
2. Applies it to TEST set (no tuning on test!)
3. Reports final metrics
4. Ensures test set remains independent
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

# =============================================================================
# LOAD OPTIMAL THRESHOLDS
# =============================================================================

def load_optimal_thresholds():
    """Load optimal thresholds from validation tuning."""
    print("=" * 70)
    print("LOADING OPTIMAL THRESHOLDS FROM VALIDATION TUNING")
    print("=" * 70)
    
    # Find most recent threshold tuning results
    threshold_files = glob.glob(os.path.join(RESULTS_DIR, "threshold_tuning_validation_*.json"))
    
    if not threshold_files:
        print("[ERROR] No validation threshold tuning results found!")
        print("[INFO] Run tune_threshold_validation.py first!")
        return None, None
    
    # Load most recent
    latest_file = max(threshold_files, key=os.path.getctime)
    print(f"Loading thresholds from: {os.path.basename(latest_file)}")
    
    with open(latest_file, 'r') as f:
        tuning_results = json.load(f)
    
    baseline_threshold = tuning_results['baseline_model']['optimal_threshold']
    autoencoder_threshold = tuning_results['autoencoder_model']['optimal_threshold']
    
    print(f"\n✅ Optimal Thresholds (from validation):")
    print(f"  Baseline Model:    {baseline_threshold:.4f}")
    print(f"  Autoencoder Model: {autoencoder_threshold:.4f}")
    
    return baseline_threshold, autoencoder_threshold

# =============================================================================
# LOAD MODELS AND TEST DATA
# =============================================================================

def load_models_and_test_data():
    """Load models and TEST sliding window features."""
    print("\n" + "=" * 70)
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
    
    # Load TEST sliding window features
    features_file = os.path.join(FEATURES_DIR, "test_sliding_features_step10.csv")
    if not os.path.exists(features_file):
        print(f"[ERROR] Test features not found: {features_file}")
        return None, None, None, None, None, None
    
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df):,} TEST feature vectors")
    
    # Check class distribution
    if 'label' in features_df.columns:
        class_dist = features_df['label'].value_counts()
        print(f"\nTest Class Distribution:")
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

def evaluate_at_threshold(model, X, y, threshold, model_name=""):
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
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        'model_name': model_name,
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

# =============================================================================
# MAIN: EVALUATE TEST SET
# =============================================================================

def main():
    """Main evaluation pipeline on test set with validation-tuned thresholds."""
    print("=" * 70)
    print("TEST SET EVALUATION WITH VALIDATION-TUNED THRESHOLDS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✅ BEST PRACTICE: Using thresholds from validation set")
    print(f"   No tuning on test set (prevents data leakage)")
    
    # Load optimal thresholds from validation
    baseline_threshold, autoencoder_threshold = load_optimal_thresholds()
    if baseline_threshold is None:
        return 1
    
    # Load models and test data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y_test, feature_cols = load_models_and_test_data()
    if baseline_model is None:
        return 1
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================================================
    # EVALUATE BASELINE MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING BASELINE MODEL ON TEST SET")
    print("=" * 70)
    print(f"Using threshold: {baseline_threshold:.4f} (from validation)")
    
    baseline_results = evaluate_at_threshold(
        baseline_model, baseline_X, y_test, baseline_threshold, "baseline"
    )
    
    print(f"\n📊 Test Set Results (Baseline):")
    print(f"  Accuracy:  {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)")
    print(f"  Precision: {baseline_results['precision']:.4f} ({baseline_results['precision']*100:.2f}%)")
    print(f"  Recall:    {baseline_results['recall']:.4f} ({baseline_results['recall']*100:.2f}%)")
    print(f"  F1-Score:  {baseline_results['f1_score']:.4f} ({baseline_results['f1_score']*100:.2f}%)")
    print(f"  ROC AUC:   {baseline_results['roc_auc']:.4f}" if baseline_results['roc_auc'] else "  ROC AUC:   N/A")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {baseline_results['tn']:8d}  {baseline_results['fp']:8d}")
    print(f"Positive  {baseline_results['fn']:8d}  {baseline_results['tp']:8d}")
    print(f"\n  Additional Metrics:")
    print(f"  False Positive Rate: {baseline_results['fpr']:.4f} ({baseline_results['fpr']*100:.2f}%)")
    print(f"  False Negative Rate: {baseline_results['fnr']:.4f} ({baseline_results['fnr']*100:.2f}%)")
    
    # ========================================================================
    # EVALUATE AUTOENCODER MODEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("EVALUATING AUTOENCODER MODEL ON TEST SET")
    print("=" * 70)
    print(f"Using threshold: {autoencoder_threshold:.4f} (from validation)")
    
    autoencoder_results = evaluate_at_threshold(
        autoencoder_model, autoencoder_X, y_test, autoencoder_threshold, "autoencoder"
    )
    
    print(f"\n📊 Test Set Results (Autoencoder):")
    print(f"  Accuracy:  {autoencoder_results['accuracy']:.4f} ({autoencoder_results['accuracy']*100:.2f}%)")
    print(f"  Precision: {autoencoder_results['precision']:.4f} ({autoencoder_results['precision']*100:.2f}%)")
    print(f"  Recall:    {autoencoder_results['recall']:.4f} ({autoencoder_results['recall']*100:.2f}%)")
    print(f"  F1-Score:  {autoencoder_results['f1_score']:.4f} ({autoencoder_results['f1_score']*100:.2f}%)")
    print(f"  ROC AUC:   {autoencoder_results['roc_auc']:.4f}" if autoencoder_results['roc_auc'] else "  ROC AUC:   N/A")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {autoencoder_results['tn']:8d}  {autoencoder_results['fp']:8d}")
    print(f"Positive  {autoencoder_results['fn']:8d}  {autoencoder_results['tp']:8d}")
    print(f"\n  Additional Metrics:")
    print(f"  False Positive Rate: {autoencoder_results['fpr']:.4f} ({autoencoder_results['fpr']*100:.2f}%)")
    print(f"  False Negative Rate: {autoencoder_results['fnr']:.4f} ({autoencoder_results['fnr']*100:.2f}%)")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING TEST SET EVALUATION RESULTS")
    print("=" * 70)
    
    results_summary = {
        'timestamp': timestamp,
        'evaluation_set': 'test',
        'thresholds_from': 'validation',
        'baseline_model': {
            'threshold': float(baseline_threshold),
            'results': baseline_results
        },
        'autoencoder_model': {
            'threshold': float(autoencoder_threshold),
            'results': autoencoder_results
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, f"test_evaluation_validation_threshold_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Saved results to: {results_file}")
    
    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Autoencoder':<15} {'Winner':<15}")
    print("-" * 65)
    print(f"{'F1-Score':<20} {baseline_results['f1_score']:<15.4f} {autoencoder_results['f1_score']:<15.4f} ", end="")
    if autoencoder_results['f1_score'] > baseline_results['f1_score']:
        print("Autoencoder ✅")
    else:
        print("Baseline ✅")
    
    print(f"{'Precision':<20} {baseline_results['precision']:<15.4f} {autoencoder_results['precision']:<15.4f} ", end="")
    if autoencoder_results['precision'] > baseline_results['precision']:
        print("Autoencoder ✅")
    else:
        print("Baseline ✅")
    
    print(f"{'Recall':<20} {baseline_results['recall']:<15.4f} {autoencoder_results['recall']:<15.4f} ", end="")
    if autoencoder_results['recall'] > baseline_results['recall']:
        print("Autoencoder ✅")
    else:
        print("Baseline ✅")
    
    print(f"{'ROC AUC':<20} ", end="")
    if baseline_results['roc_auc']:
        print(f"{baseline_results['roc_auc']:<15.4f} ", end="")
    else:
        print(f"{'N/A':<15} ", end="")
    if autoencoder_results['roc_auc']:
        print(f"{autoencoder_results['roc_auc']:<15.4f} ", end="")
    else:
        print(f"{'N/A':<15} ", end="")
    if autoencoder_results['roc_auc'] and baseline_results['roc_auc']:
        if autoencoder_results['roc_auc'] > baseline_results['roc_auc']:
            print("Autoencoder ✅")
        else:
            print("Baseline ✅")
    else:
        print("N/A")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print("\n✅ Test set evaluated with validation-tuned thresholds")
    print("✅ No data leakage (thresholds from validation, not test)")
    print("✅ Results are reliable and unbiased")
    
    return 0

if __name__ == "__main__":
    exit(main())





