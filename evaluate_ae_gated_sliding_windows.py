#!/usr/bin/env python3
"""
Evaluate AE-Gated Model on Sliding Windows
===========================================

Evaluates the AE-gated RF model on sliding window test set with multiple thresholds
and compares to baseline sliding window results.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for sliding window evaluation."""
    
    # Model file (latest AE-gated model)
    MODEL_PATTERN = "rf_ae_gated_ml_*.pkl"
    
    # Test data
    TEST_SLIDING_FEATURES = "datasets/test_sliding_features.csv"
    
    # Baseline metrics (from sliding window evaluation)
    BASELINE_METRICS = {
        'threshold_0.45': {
            'precision': 0.0165,
            'recall': 0.4263,
            'f1_score': 0.0318
        },
        'threshold_0.60': {
            'precision': 0.0235,
            'recall': 0.2184,
            'f1_score': 0.0425
        },
        'threshold_0.75': {
            'precision': 0.0286,
            'recall': 0.1342,
            'f1_score': 0.0472
        },
        'threshold_0.90': {
            'precision': 0.0378,
            'recall': 0.0658,
            'f1_score': 0.0480,
            'roc_auc': 0.7457
        }
    }
    
    # Thresholds to test
    THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    # Output directories
    RESULTS_DIR = "results"
    
    # Feature columns to exclude
    EXCLUDE_COLUMNS = ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label',
                       'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx',
                       'sliding_start_sclk', 'sliding_end_sclk']

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

def load_ae_gated_model():
    """Load the latest AE-gated model."""
    print("=" * 70)
    print("LOADING AE-GATED MODEL")
    print("=" * 70)
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"[ERROR] Models directory not found: {models_dir}")
        return None, None
    
    # Find all AE-gated models
    import glob
    model_files = glob.glob(os.path.join(models_dir, Config.MODEL_PATTERN))
    
    if not model_files:
        print(f"[ERROR] No AE-gated model found matching: {Config.MODEL_PATTERN}")
        return None, None
    
    # Get latest model
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading model: {os.path.basename(latest_model)}")
    
    model = joblib.load(latest_model)
    print(f"Model loaded successfully")
    
    return model, latest_model

def load_test_sliding_features():
    """Load sliding window test features."""
    print("\n" + "=" * 70)
    print("LOADING SLIDING WINDOW TEST FEATURES")
    print("=" * 70)
    
    if not os.path.exists(Config.TEST_SLIDING_FEATURES):
        print(f"[ERROR] Test sliding features file not found: {Config.TEST_SLIDING_FEATURES}")
        return None, None, None
    
    test_df = pd.read_csv(Config.TEST_SLIDING_FEATURES)
    print(f"Loaded {len(test_df):,} sliding window feature vectors")
    
    # Get feature columns
    feature_cols = [col for col in test_df.columns 
                   if col not in Config.EXCLUDE_COLUMNS]
    
    print(f"Features: {len(feature_cols)}")
    
    # Filter out 'Omit' labels and convert to binary
    valid_mask = test_df['label'] != 'Omit'
    valid_df = test_df[valid_mask].copy()
    
    # Convert labels to binary (True -> 1, False -> 0)
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    # Prepare data
    X = valid_df[feature_cols].values
    y = valid_df['label'].values.astype(int)
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\nClass distribution (after filtering 'Omit'):")
    for label, count in sorted(class_dist.items()):
        pct = count / len(y) * 100
        label_name = 'Positive' if label == 1 else 'Negative'
        print(f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")
    
    print(f"\nTotal valid samples: {len(y):,}")
    print(f"Omitted samples: {(~valid_mask).sum():,}")
    
    return X, y, feature_cols, valid_df

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_at_threshold(model, X, y, threshold, feature_cols):
    """Evaluate model at a specific threshold."""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Make predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Check if any positive predictions
    if np.sum(y_pred) == 0:
        return {
            'threshold': threshold,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': int(np.sum(y == 1)),
            'tn': int(np.sum(y == 0)),
            'predictions': 0
        }
    
    # Calculate metrics
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        if len(cm) == 1:
            if y_pred.sum() == 0:
                tn, fp, fn, tp = int(cm[0, 0]), 0, int(y.sum()), 0
            else:
                tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    
    return {
        'threshold': threshold,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'predictions': int(np.sum(y_pred))
    }

def evaluate_multiple_thresholds(model, X, y, feature_cols):
    """Evaluate model at multiple thresholds."""
    print("\n" + "=" * 70)
    print("EVALUATING AT MULTIPLE THRESHOLDS")
    print("=" * 70)
    
    results = []
    
    # Get probabilities once
    y_proba = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_proba)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8}")
    print("-" * 70)
    
    for threshold in Config.THRESHOLDS:
        result = evaluate_at_threshold(model, X, y, threshold, feature_cols)
        result['roc_auc'] = float(roc_auc)  # Same for all thresholds
        results.append(result)
        
        print(f"{threshold:<12.2f} {result['precision']:<12.4f} {result['recall']:<12.4f} "
              f"{result['f1_score']:<12.4f} {result['tp']:<8} {result['fp']:<8}")
    
    # Find best F1 threshold
    best_f1_idx = np.argmax([r['f1_score'] for r in results])
    best_result = results[best_f1_idx]
    
    print("\n" + "=" * 70)
    print("BEST PERFORMANCE (BY F1-SCORE)")
    print("=" * 70)
    print(f"Threshold: {best_result['threshold']:.2f}")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall:    {best_result['recall']:.4f}")
    print(f"F1-Score:  {best_result['f1_score']:.4f}")
    print(f"ROC AUC:   {best_result['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {best_result['tn']:8d}  {best_result['fp']:8d}")
    print(f"Positive  {best_result['fn']:8d}  {best_result['tp']:8d}")
    
    return results, best_result, roc_auc

# =============================================================================
# COMPARISON TO BASELINE
# =============================================================================

def compare_to_baseline(new_results, best_result, roc_auc):
    """Compare new results to baseline."""
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (SLIDING WINDOWS)")
    print("=" * 70)
    
    # Baseline at threshold 0.90 (best baseline)
    baseline = Config.BASELINE_METRICS['threshold_0.90']
    
    print(f"\nBaseline (threshold=0.90):")
    print(f"  Precision: {baseline['precision']:.4f}")
    print(f"  Recall:    {baseline.get('recall', 'N/A')}")
    print(f"  F1-Score:  {baseline['f1_score']:.4f}")
    print(f"  ROC AUC:   {baseline.get('roc_auc', 0.7457):.4f}")
    
    print(f"\nNew Model (threshold={best_result['threshold']:.2f}):")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall:    {best_result['recall']:.4f}")
    print(f"  F1-Score:  {best_result['f1_score']:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    print(f"\n{'Metric':<15} {'Baseline':<12} {'New Model':<12} {'Change':<12} {'Status':<10}")
    print("-" * 70)
    
    # Compare precision
    precision_delta = best_result['precision'] - baseline['precision']
    precision_pct = (precision_delta / baseline['precision'] * 100) if baseline['precision'] > 0 else 0
    precision_status = "[IMPROVED]" if precision_delta > 0 else "[WORSE]"
    print(f"{'Precision':<15} {baseline['precision']:<12.4f} {best_result['precision']:<12.4f} "
          f"{precision_delta:+.4f} ({precision_pct:+.1f}%)  {precision_status}")
    
    # Compare recall
    baseline_recall = baseline.get('recall', 0.0658)
    recall_delta = best_result['recall'] - baseline_recall
    recall_pct = (recall_delta / baseline_recall * 100) if baseline_recall > 0 else 0
    recall_status = "[IMPROVED]" if recall_delta > 0 else "[WORSE]"
    print(f"{'Recall':<15} {baseline_recall:<12.4f} {best_result['recall']:<12.4f} "
          f"{recall_delta:+.4f} ({recall_pct:+.1f}%)  {recall_status}")
    
    # Compare F1
    f1_delta = best_result['f1_score'] - baseline['f1_score']
    f1_pct = (f1_delta / baseline['f1_score'] * 100) if baseline['f1_score'] > 0 else 0
    f1_status = "[IMPROVED]" if f1_delta > 0 else "[WORSE]"
    print(f"{'F1-Score':<15} {baseline['f1_score']:<12.4f} {best_result['f1_score']:<12.4f} "
          f"{f1_delta:+.4f} ({f1_pct:+.1f}%)  {f1_status}")
    
    # Compare ROC AUC
    baseline_auc = baseline.get('roc_auc', 0.7457)
    auc_delta = roc_auc - baseline_auc
    auc_pct = (auc_delta / baseline_auc * 100) if baseline_auc > 0 else 0
    auc_status = "[IMPROVED]" if auc_delta > 0 else "[WORSE]"
    print(f"{'ROC AUC':<15} {baseline_auc:<12.4f} {roc_auc:<12.4f} "
          f"{auc_delta:+.4f} ({auc_pct:+.1f}%)  {auc_status}")
    
    # Overall verdict
    print("\n" + "=" * 70)
    if best_result['f1_score'] > baseline['f1_score']:
        print("[SUCCESS] OVERALL: IMPROVEMENT DETECTED")
    else:
        print("[FAILED] OVERALL: NO IMPROVEMENT (or worse)")
    print("=" * 70)
    
    return {
        'precision_delta': precision_delta,
        'recall_delta': recall_delta,
        'f1_delta': f1_delta,
        'roc_auc_delta': auc_delta
    }

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(all_results, best_result, roc_auc, comparison, model_path):
    """Save evaluation results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'experiment': 'AE-Gated Model - Sliding Window Evaluation',
        'model_file': os.path.basename(model_path),
        'test_file': Config.TEST_SLIDING_FEATURES,
        'baseline_metrics': Config.BASELINE_METRICS,
        'all_thresholds': all_results,
        'best_performance': best_result,
        'roc_auc': float(roc_auc),
        'comparison_to_baseline': comparison
    }
    
    results_path = os.path.join(Config.RESULTS_DIR, f"ae_gated_sliding_evaluation_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results_path

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("AE-GATED MODEL - SLIDING WINDOW EVALUATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Load model
    model, model_path = load_ae_gated_model()
    if model is None:
        return 1
    
    # Step 2: Load test data
    X, y, feature_cols, test_df = load_test_sliding_features()
    if X is None:
        return 1
    
    # Step 3: Evaluate at multiple thresholds
    all_results, best_result, roc_auc = evaluate_multiple_thresholds(model, X, y, feature_cols)
    
    # Step 4: Compare to baseline
    comparison = compare_to_baseline(all_results, best_result, roc_auc)
    
    # Step 5: Save results
    results_path = save_results(all_results, best_result, roc_auc, comparison, model_path)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Best F1-Score: {best_result['f1_score']:.4f} (at threshold {best_result['threshold']:.2f})")
    print(f"  Baseline F1:   {Config.BASELINE_METRICS['threshold_0.90']['f1_score']:.4f}")
    print(f"  Improvement:   {comparison['f1_delta']:+.4f}")
    print(f"\nResults saved to: {results_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
