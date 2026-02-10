#!/usr/bin/env python3
"""
Sliding Window Evaluation - Comprehensive Dataset
=================================================

As a seasoned RF scientist, this script:
1. Evaluates both models (baseline and autoencoder) on sliding windows
2. Optimizes decision thresholds for deployment scenario
3. Compares fixed vs sliding window performance
4. Provides deployment-ready metrics

This enables realistic deployment simulation evaluation.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models on sliding windows and optimize thresholds',
        epilog="""
Examples:
  python evaluate_sliding_windows.py --split val --step_size 10
  python evaluate_sliding_windows.py --split test --step_size 10
        """
    )
    
    parser.add_argument('--split', 
                       choices=['val', 'test'],
                       required=True,
                       help='Which temporal split to process')
    
    parser.add_argument('--step_size',
                       type=int,
                       default=10,
                       help='Step size for sliding windows (default: 10)')
    
    parser.add_argument('--optimize_threshold',
                       action='store_true',
                       help='Optimize decision threshold for deployment')
    
    return parser.parse_args()

# =============================================================================
# LOAD MODELS
# =============================================================================

def load_latest_models():
    """Load the most recent baseline and autoencoder models."""
    print("=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    
    # Find latest baseline model
    baseline_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_model_*.pkl"))
    if not baseline_files:
        print("[ERROR] No baseline model found!")
        return None, None, None, None
    
    baseline_file = max(baseline_files, key=os.path.getctime)
    print(f"Loading baseline model: {os.path.basename(baseline_file)}")
    baseline_model = joblib.load(baseline_file)
    
    # Load baseline metadata to get features
    baseline_meta_file = baseline_file.replace('.pkl', '_metadata.json').replace('baseline_rf_model_', 'baseline_rf_metadata_')
    if os.path.exists(baseline_meta_file):
        with open(baseline_meta_file, 'r') as f:
            baseline_metadata = json.load(f)
            baseline_features = baseline_metadata.get('features', [])
    else:
        baseline_features = None
    
    # Find latest autoencoder model
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    if not autoencoder_files:
        print("[ERROR] No autoencoder model found!")
        return baseline_model, baseline_features, None, None
    
    autoencoder_file = max(autoencoder_files, key=os.path.getctime)
    print(f"Loading autoencoder model: {os.path.basename(autoencoder_file)}")
    autoencoder_model = joblib.load(autoencoder_file)
    
    # Load autoencoder metadata to get features
    autoencoder_meta_file = autoencoder_file.replace('.pkl', '_metadata.json').replace('rf_with_autoencoder_', 'rf_with_autoencoder_metadata_')
    if os.path.exists(autoencoder_meta_file):
        with open(autoencoder_meta_file, 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = None
    
    print(f"Baseline features: {len(baseline_features) if baseline_features else 'Unknown'}")
    print(f"Autoencoder features: {len(autoencoder_features) if autoencoder_features else 'Unknown'}")
    
    return baseline_model, baseline_features, autoencoder_model, autoencoder_features

# =============================================================================
# LOAD SLIDING WINDOW FEATURES
# =============================================================================

def load_sliding_features(split_name, step_size):
    """Load sliding window features."""
    print(f"\nLoading sliding window features for {split_name} split...")
    
    sliding_features_file = os.path.join(FEATURES_DIR, f"{split_name}_sliding_features_step{step_size}.csv")
    
    if not os.path.exists(sliding_features_file):
        print(f"[ERROR] Sliding features file not found: {sliding_features_file}")
        print("[INFO] Run engineer_sliding_features.py first!")
        return None, None
    
    features_df = pd.read_csv(sliding_features_file)
    print(f"  Loaded {len(features_df):,} feature vectors")
    
    # Check class distribution
    if 'label' in features_df.columns:
        class_dist = features_df['label'].value_counts()
        total = len(features_df)
        print(f"  Class distribution:")
        print(f"    Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total*100:.2f}%)")
        print(f"    Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total*100:.2f}%)")
        if class_dist.get(1, 0) > 0:
            ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
            print(f"    Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    # Separate features and labels
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    X = features_df[feature_cols].values
    y = features_df['label'].values if 'label' in features_df.columns else None
    
    return X, y, feature_cols, features_df

# =============================================================================
# EVALUATE MODEL
# =============================================================================

def evaluate_model(model, X, y, feature_cols, threshold=0.5, model_name=""):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print(f"EVALUATION - {model_name.upper()} (Threshold: {threshold:.2f})")
    print("=" * 70)
    
    # Ensure we have the right features
    available_features = [f for f in feature_cols if f in pd.DataFrame(X, columns=feature_cols).columns]
    if len(available_features) != len(feature_cols):
        print(f"[WARNING] Feature mismatch: {len(available_features)} vs {len(feature_cols)}")
        X_eval = pd.DataFrame(X, columns=feature_cols)[available_features].values
    else:
        X_eval = X
    
    # Get probabilities
    y_proba = model.predict_proba(X_eval)[:, 1]
    
    # Predictions with custom threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Metrics
    accuracy = (y_pred == y).mean()
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y, y_proba)
    except ValueError:
        roc_auc = np.nan
        print("[WARNING] ROC AUC cannot be calculated (only one class present)")
    
    # Average Precision (PR AUC)
    try:
        pr_auc = average_precision_score(y, y_proba)
    except ValueError:
        pr_auc = np.nan
    
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
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    print(f"Performance Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}" if not np.isnan(roc_auc) else "  ROC AUC:   N/A")
    print(f"  PR AUC:    {pr_auc:.4f}" if not np.isnan(pr_auc) else "  PR AUC:    N/A")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {tn:8d}  {fp:8d}")
    print(f"Positive  {fn:8d}  {tp:8d}")
    print(f"\nAdditional Metrics:")
    print(f"  True Positive Rate (TPR):  {tpr:.4f}")
    print(f"  False Positive Rate (FPR): {fpr:.4f}")
    print(f"  False Negative Rate (FNR): {fnr:.4f}")
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'pr_auc': float(pr_auc) if not np.isnan(pr_auc) else None,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'tpr': float(tpr),
        'fpr': float(fpr),
        'fnr': float(fnr)
    }

# =============================================================================
# OPTIMIZE THRESHOLD
# =============================================================================

def optimize_threshold(y_true, y_proba, metric='f1', target_precision=None):
    """Optimize decision threshold."""
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 70)
    
    # Test thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal threshold based on metric
    if target_precision is not None:
        # Find threshold that achieves target precision
        valid_results = results_df[results_df['precision'] >= target_precision]
        if len(valid_results) > 0:
            optimal_idx = valid_results['f1_score'].idxmax()
        else:
            optimal_idx = results_df['precision'].idxmax()
    elif metric == 'f1':
        optimal_idx = results_df['f1_score'].idxmax()
    elif metric == 'precision':
        optimal_idx = results_df['precision'].idxmax()
    elif metric == 'recall':
        optimal_idx = results_df['recall'].idxmax()
    else:
        optimal_idx = results_df['f1_score'].idxmax()
    
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    optimal_metrics = results_df.loc[optimal_idx]
    
    print(f"Optimal threshold: {optimal_threshold:.3f} (metric: {metric})")
    print(f"  Precision: {optimal_metrics['precision']:.4f}")
    print(f"  Recall:    {optimal_metrics['recall']:.4f}")
    print(f"  F1-Score:  {optimal_metrics['f1_score']:.4f}")
    print(f"  FPR:       {optimal_metrics['fpr']:.4f}")
    
    return optimal_threshold, results_df

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(baseline_metrics, autoencoder_metrics, split_name, step_size):
    """Save evaluation results."""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'split': split_name,
        'step_size': step_size,
        'evaluation_type': 'sliding_windows',
        'baseline_model': baseline_metrics,
        'autoencoder_model': autoencoder_metrics,
        'comparison': {
            'accuracy_diff': (autoencoder_metrics['accuracy'] - baseline_metrics['accuracy']) if autoencoder_metrics else None,
            'precision_diff': (autoencoder_metrics['precision'] - baseline_metrics['precision']) if autoencoder_metrics else None,
            'recall_diff': (autoencoder_metrics['recall'] - baseline_metrics['recall']) if autoencoder_metrics else None,
            'f1_diff': (autoencoder_metrics['f1_score'] - baseline_metrics['f1_score']) if autoencoder_metrics else None,
            'roc_auc_diff': ((autoencoder_metrics['roc_auc'] or 0) - (baseline_metrics['roc_auc'] or 0)) if autoencoder_metrics else None
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, f"{split_name}_sliding_evaluation_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return results_file

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main evaluation pipeline."""
    args = parse_arguments()
    
    print("=" * 70)
    print("SLIDING WINDOW EVALUATION - COMPREHENSIVE DATASET")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Step size: {args.step_size}")
    print(f"Optimize threshold: {args.optimize_threshold}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models
    baseline_model, baseline_features, autoencoder_model, autoencoder_features = load_latest_models()
    
    if baseline_model is None:
        return 1
    
    # Load sliding window features
    X, y, feature_cols, features_df = load_sliding_features(args.split, args.step_size)
    
    if X is None or y is None:
        return 1
    
    # Select features for baseline model (original 15)
    if baseline_features:
        baseline_feature_cols = [f for f in baseline_features if f in feature_cols]
        baseline_X = features_df[baseline_feature_cols].values
    else:
        # Default to first 15 features
        baseline_X = X[:, :15]
        baseline_feature_cols = feature_cols[:15]
    
    # Select features for autoencoder model (all features)
    if autoencoder_model and autoencoder_features:
        autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
        autoencoder_X = features_df[autoencoder_feature_cols].values
    else:
        autoencoder_X = X
        autoencoder_feature_cols = feature_cols
    
    # Default threshold
    threshold = 0.5
    
    # Optimize threshold if requested
    if args.optimize_threshold and baseline_model is not None:
        print("\nOptimizing threshold for baseline model...")
        baseline_proba = baseline_model.predict_proba(baseline_X)[:, 1]
        optimal_threshold, threshold_results = optimize_threshold(y, baseline_proba, metric='f1')
        threshold = optimal_threshold
        print(f"\nUsing optimized threshold: {threshold:.3f}")
    
    # Evaluate baseline model
    baseline_metrics = evaluate_model(
        baseline_model, baseline_X, y, baseline_feature_cols, 
        threshold=threshold, model_name="Baseline Model"
    )
    
    # Evaluate autoencoder model (if available)
    autoencoder_metrics = None
    if autoencoder_model is not None:
        autoencoder_metrics = evaluate_model(
            autoencoder_model, autoencoder_X, y, autoencoder_feature_cols,
            threshold=threshold, model_name="Autoencoder Model"
        )
    
    # Compare models
    if autoencoder_metrics:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<20} {'Baseline':<15} {'Autoencoder':<15} {'Difference':<15}")
        print("-" * 70)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            baseline_val = baseline_metrics.get(metric, 0) or 0
            autoencoder_val = autoencoder_metrics.get(metric, 0) or 0
            diff = autoencoder_val - baseline_val
            diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A"
            
            baseline_str = f"{baseline_val:.4f}" if not np.isnan(baseline_val) else "N/A"
            autoencoder_str = f"{autoencoder_val:.4f}" if not np.isnan(autoencoder_val) else "N/A"
            
            print(f"{metric:<20} {baseline_str:<15} {autoencoder_str:<15} {diff_str:<15}")
    
    # Save results
    results_file = save_results(baseline_metrics, autoencoder_metrics, args.split, args.step_size)
    
    print("\n" + "=" * 70)
    print("SLIDING WINDOW EVALUATION COMPLETED")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review deployment metrics (realistic imbalance)")
    print(f"  2. Optimize threshold for deployment requirements")
    print(f"  3. Compare with fixed window performance")
    
    return 0

if __name__ == "__main__":
    exit(main())

