#!/usr/bin/env python3
"""
Compare Comprehensive Models with Original Thresholds
=====================================================

Evaluates both baseline and autoencoder models on comprehensive dataset
at the same thresholds used in the original ml_ready_vortex_data.csv evaluation
(0.45, 0.60, 0.75, 0.90) for direct comparison.
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, accuracy_score
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

# Original thresholds from ml_ready_vortex_data.csv evaluation
ORIGINAL_THRESHOLDS = [0.45, 0.60, 0.75, 0.90]

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

def load_models_and_data(split_name, step_size):
    """Load models and sliding window features."""
    print("=" * 70)
    print("LOADING MODELS AND DATA")
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
    
    # Load sliding window features
    features_file = os.path.join(FEATURES_DIR, f"{split_name}_sliding_features_step{step_size}.csv")
    features_df = pd.read_csv(features_file)
    
    print(f"Loaded {len(features_df):,} feature vectors")
    
    # Separate features and labels
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Baseline uses first 15 features, autoencoder uses all
    baseline_X = X[:, :15]
    autoencoder_X = X
    
    return baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, feature_cols

# =============================================================================
# EVALUATE AT THRESHOLD
# =============================================================================

def evaluate_at_threshold(model, X, y, threshold, model_name=""):
    """Evaluate model at specific threshold."""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    
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
# MAIN COMPARISON
# =============================================================================

def main():
    """Main comparison pipeline."""
    parser = argparse.ArgumentParser(description='Compare comprehensive models with original thresholds')
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Split to evaluate')
    parser.add_argument('--step_size', type=int, default=10, help='Step size for sliding windows')
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPARING COMPREHENSIVE MODELS WITH ORIGINAL THRESHOLDS")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Step size: {args.step_size}")
    print(f"Thresholds: {ORIGINAL_THRESHOLDS}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models and data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, feature_cols = load_models_and_data(
        args.split, args.step_size
    )
    
    if baseline_model is None:
        return 1
    
    # Class distribution
    class_dist = pd.Series(y).value_counts()
    total = len(y)
    print(f"\nClass Distribution:")
    print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total*100:.2f}%)")
    print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total*100:.2f}%)")
    if class_dist.get(1, 0) > 0:
        ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
        print(f"  Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    # Evaluate at each threshold
    print("\n" + "=" * 70)
    print("EVALUATION AT ORIGINAL THRESHOLDS")
    print("=" * 70)
    
    baseline_results = []
    autoencoder_results = []
    
    for threshold in ORIGINAL_THRESHOLDS:
        print(f"\nThreshold: {threshold:.2f}")
        print("-" * 70)
        
        # Baseline model
        baseline_metrics = evaluate_at_threshold(baseline_model, baseline_X, y, threshold, "Baseline")
        baseline_results.append(baseline_metrics)
        
        print(f"Baseline Model:")
        print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f} ({baseline_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {baseline_metrics['precision']:.4f} ({baseline_metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {baseline_metrics['recall']:.4f} ({baseline_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {baseline_metrics['f1_score']:.4f} ({baseline_metrics['f1_score']*100:.2f}%)")
        print(f"  TP: {baseline_metrics['tp']}, FP: {baseline_metrics['fp']}")
        
        # Autoencoder model
        autoencoder_metrics = evaluate_at_threshold(autoencoder_model, autoencoder_X, y, threshold, "Autoencoder")
        autoencoder_results.append(autoencoder_metrics)
        
        print(f"\nAutoencoder Model:")
        print(f"  Accuracy:  {autoencoder_metrics['accuracy']:.4f} ({autoencoder_metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {autoencoder_metrics['precision']:.4f} ({autoencoder_metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {autoencoder_metrics['recall']:.4f} ({autoencoder_metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {autoencoder_metrics['f1_score']:.4f} ({autoencoder_metrics['f1_score']*100:.2f}%)")
        print(f"  TP: {autoencoder_metrics['tp']}, FP: {autoencoder_metrics['fp']}")
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE - BASELINE MODEL")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<10}")
    print("-" * 70)
    for r in baseline_results:
        print(f"{r['threshold']:<12.2f} {r['accuracy']*100:<12.2f} {r['precision']*100:<12.2f} {r['recall']*100:<12.2f} {r['f1_score']*100:<12.2f} {r['tp']:<8} {r['fp']:<10}")
    
    print("\n" + "=" * 70)
    print("COMPARISON TABLE - AUTOENCODER MODEL")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<10}")
    print("-" * 70)
    for r in autoencoder_results:
        print(f"{r['threshold']:<12.2f} {r['accuracy']*100:<12.2f} {r['precision']*100:<12.2f} {r['recall']*100:<12.2f} {r['f1_score']*100:<12.2f} {r['tp']:<8} {r['fp']:<10}")
    
    # Compare with original
    print("\n" + "=" * 70)
    print("COMPARISON WITH ORIGINAL MODEL (ml_ready_vortex_data.csv)")
    print("=" * 70)
    
    # Original results (from image/description)
    original_results = [
        {'threshold': 0.45, 'precision': 0.0165, 'recall': 0.4263, 'f1_score': 0.0318, 'tp': 162, 'fp': 9642},
        {'threshold': 0.60, 'precision': 0.0235, 'recall': 0.2184, 'f1_score': 0.0425, 'tp': 83, 'fp': 3445},
        {'threshold': 0.75, 'precision': 0.0286, 'recall': 0.1342, 'f1_score': 0.0472, 'tp': 51, 'fp': 1731},
        {'threshold': 0.90, 'precision': 0.0378, 'recall': 0.0658, 'f1_score': 0.0480, 'tp': 25, 'fp': 636}
    ]
    
    print(f"\n{'Threshold':<12} {'Model':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<10}")
    print("-" * 100)
    
    for i, threshold in enumerate(ORIGINAL_THRESHOLDS):
        orig = original_results[i]
        base = baseline_results[i]
        auto = autoencoder_results[i]
        
        print(f"{threshold:<12.2f} {'Original':<20} {orig['precision']*100:<12.2f} {orig['recall']*100:<12.2f} {orig['f1_score']*100:<12.2f} {orig['tp']:<8} {orig['fp']:<10}")
        print(f"{'':<12} {'Baseline':<20} {base['precision']*100:<12.2f} {base['recall']*100:<12.2f} {base['f1_score']*100:<12.2f} {base['tp']:<8} {base['fp']:<10}")
        print(f"{'':<12} {'Autoencoder':<20} {auto['precision']*100:<12.2f} {auto['recall']*100:<12.2f} {auto['f1_score']*100:<12.2f} {auto['tp']:<8} {auto['fp']:<10}")
        print()
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    # Find best F1 for each model
    orig_best = max(original_results, key=lambda x: x['f1_score'])
    base_best = max(baseline_results, key=lambda x: x['f1_score'])
    auto_best = max(autoencoder_results, key=lambda x: x['f1_score'])
    
    print(f"\nBest F1-Score:")
    print(f"  Original:     {orig_best['f1_score']*100:.2f}% at threshold {orig_best['threshold']:.2f}")
    print(f"  Baseline:     {base_best['f1_score']*100:.2f}% at threshold {base_best['threshold']:.2f}")
    print(f"  Autoencoder:  {auto_best['f1_score']*100:.2f}% at threshold {auto_best['threshold']:.2f}")
    
    print(f"\nHighest Precision:")
    orig_max_prec = max(original_results, key=lambda x: x['precision'])
    base_max_prec = max(baseline_results, key=lambda x: x['precision'])
    auto_max_prec = max(autoencoder_results, key=lambda x: x['precision'])
    print(f"  Original:     {orig_max_prec['precision']*100:.2f}% at threshold {orig_max_prec['threshold']:.2f}")
    print(f"  Baseline:     {base_max_prec['precision']*100:.2f}% at threshold {base_max_prec['threshold']:.2f}")
    print(f"  Autoencoder:  {auto_max_prec['precision']*100:.2f}% at threshold {auto_max_prec['threshold']:.2f}")
    
    print(f"\nHighest Recall:")
    orig_max_recall = max(original_results, key=lambda x: x['recall'])
    base_max_recall = max(baseline_results, key=lambda x: x['recall'])
    auto_max_recall = max(autoencoder_results, key=lambda x: x['recall'])
    print(f"  Original:     {orig_max_recall['recall']*100:.2f}% at threshold {orig_max_recall['threshold']:.2f}")
    print(f"  Baseline:     {base_max_recall['recall']*100:.2f}% at threshold {base_max_recall['threshold']:.2f}")
    print(f"  Autoencoder:  {auto_max_recall['recall']*100:.2f}% at threshold {auto_max_recall['threshold']:.2f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"{args.split}_original_thresholds_comparison_{timestamp}.json")
    
    results = {
        'timestamp': timestamp,
        'split': args.split,
        'step_size': args.step_size,
        'thresholds': ORIGINAL_THRESHOLDS,
        'class_distribution': {
            'positive': int(class_dist.get(1, 0)),
            'negative': int(class_dist.get(0, 0)),
            'ratio': float(ratio) if class_dist.get(1, 0) > 0 else None
        },
        'original_model': original_results,
        'baseline_model': baseline_results,
        'autoencoder_model': autoencoder_results,
        'comparison': {
            'best_f1': {
                'original': orig_best,
                'baseline': base_best,
                'autoencoder': auto_best
            },
            'max_precision': {
                'original': orig_max_prec,
                'baseline': base_max_prec,
                'autoencoder': auto_max_prec
            },
            'max_recall': {
                'original': orig_max_recall,
                'baseline': base_max_recall,
                'autoencoder': auto_max_recall
            }
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETED")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())

