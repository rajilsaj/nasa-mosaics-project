#!/usr/bin/env python3
"""
Fair Threshold Comparison: Original vs Comprehensive
====================================================

Compares models at their respective optimal thresholds:
- Original model: thresholds 0.45, 0.60, 0.75, 0.90 (as shown in results)
- Comprehensive models: lower thresholds (0.01-0.10) where they actually make predictions

This provides a fair comparison of model capabilities.
"""

import os
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

# Original thresholds (from ml_ready_vortex_data.csv)
ORIGINAL_THRESHOLDS = [0.45, 0.60, 0.75, 0.90]

# Comprehensive thresholds (where models actually make predictions)
COMPREHENSIVE_THRESHOLDS = [0.01, 0.02, 0.05, 0.10]

# =============================================================================
# LOAD AND EVALUATE
# =============================================================================

def evaluate_at_threshold(model, X, y, threshold):
    """Evaluate model at specific threshold."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    
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

def main():
    """Main comparison pipeline."""
    print("=" * 70)
    print("FAIR THRESHOLD COMPARISON: ORIGINAL VS COMPREHENSIVE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load comprehensive models and data
    baseline_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_model_*.pkl"))
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    
    baseline_model = joblib.load(max(baseline_files, key=os.path.getctime))
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    
    features_df = pd.read_csv(os.path.join(FEATURES_DIR, "test_sliding_features_step10.csv"))
    
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    baseline_X = features_df.iloc[:, :15].values
    autoencoder_X = features_df[feature_cols].values
    y = features_df['label'].values
    
    print(f"\nTest Set: {len(features_df):,} samples")
    print(f"  Positive: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    print(f"  Negative: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.2f}%)")
    
    # Original results (from ml_ready_vortex_data.csv)
    original_results = [
        {'threshold': 0.45, 'precision': 0.0165, 'recall': 0.4263, 'f1_score': 0.0318, 'tp': 162, 'fp': 9642, 'roc_auc': 0.7457},
        {'threshold': 0.60, 'precision': 0.0235, 'recall': 0.2184, 'f1_score': 0.0425, 'tp': 83, 'fp': 3445, 'roc_auc': 0.7457},
        {'threshold': 0.75, 'precision': 0.0286, 'recall': 0.1342, 'f1_score': 0.0472, 'tp': 51, 'fp': 1731, 'roc_auc': 0.7457},
        {'threshold': 0.90, 'precision': 0.0378, 'recall': 0.0658, 'f1_score': 0.0480, 'tp': 25, 'fp': 636, 'roc_auc': 0.7457}
    ]
    
    # Evaluate comprehensive models at their thresholds
    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODELS AT THEIR OPTIMAL THRESHOLDS")
    print("=" * 70)
    
    baseline_results = [evaluate_at_threshold(baseline_model, baseline_X, y, t) for t in COMPREHENSIVE_THRESHOLDS]
    autoencoder_results = [evaluate_at_threshold(autoencoder_model, autoencoder_X, y, t) for t in COMPREHENSIVE_THRESHOLDS]
    
    # Create comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Model':<20} {'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12} {'TP':<8} {'FP':<10}")
    print("-" * 110)
    
    # Original model results
    for orig in original_results:
        print(f"{'Original':<20} {orig['threshold']:<12.2f} {orig['precision']*100:<12.2f} {orig['recall']*100:<12.2f} {orig['f1_score']*100:<12.2f} {orig['roc_auc']*100:<12.2f} {orig['tp']:<8} {orig['fp']:<10}")
    
    print()
    
    # Comprehensive baseline results
    for base in baseline_results:
        roc_str = f"{base['roc_auc']*100:.2f}" if base['roc_auc'] else "N/A"
        print(f"{'Baseline':<20} {base['threshold']:<12.2f} {base['precision']*100:<12.2f} {base['recall']*100:<12.2f} {base['f1_score']*100:<12.2f} {roc_str:<12} {base['tp']:<8} {base['fp']:<10}")
    
    print()
    
    # Comprehensive autoencoder results
    for auto in autoencoder_results:
        roc_str = f"{auto['roc_auc']*100:.2f}" if auto['roc_auc'] else "N/A"
        print(f"{'Autoencoder':<20} {auto['threshold']:<12.2f} {auto['precision']*100:<12.2f} {auto['recall']*100:<12.2f} {auto['f1_score']*100:<12.2f} {roc_str:<12} {auto['tp']:<8} {auto['fp']:<10}")
    
    # Find best performance for each model
    print("\n" + "=" * 70)
    print("BEST PERFORMANCE COMPARISON")
    print("=" * 70)
    
    orig_best = max(original_results, key=lambda x: x['f1_score'])
    base_best = max(baseline_results, key=lambda x: x['f1_score'])
    auto_best = max(autoencoder_results, key=lambda x: x['f1_score'])
    
    print(f"\n{'Metric':<20} {'Original':<20} {'Baseline':<20} {'Autoencoder':<20} {'Winner':<20}")
    print("-" * 100)
    
    metrics = [
        ('F1-Score', 'f1_score', lambda x: x*100),
        ('Precision', 'precision', lambda x: x*100),
        ('Recall', 'recall', lambda x: x*100),
        ('ROC AUC', 'roc_auc', lambda x: x*100 if x else 0)
    ]
    
    for metric_name, metric_key, formatter in metrics:
        orig_val = formatter(orig_best[metric_key])
        base_val = formatter(base_best[metric_key])
        auto_val = formatter(auto_best[metric_key])
        
        if metric_key == 'roc_auc':
            orig_val = orig_best['roc_auc'] * 100 if orig_best['roc_auc'] else 0
            base_val = base_best['roc_auc'] * 100 if base_best['roc_auc'] else 0
            auto_val = auto_best['roc_auc'] * 100 if auto_best['roc_auc'] else 0
        
        winner = "Original" if orig_val >= max(base_val, auto_val) else ("Baseline" if base_val >= auto_val else "Autoencoder")
        
        print(f"{metric_name:<20} {orig_val:<20.2f} {base_val:<20.2f} {auto_val:<20.2f} {winner:<20}")
    
    # Detailed comparison
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)
    
    print(f"\nOriginal Model (Best F1 at threshold {orig_best['threshold']:.2f}):")
    print(f"  Precision: {orig_best['precision']*100:.2f}%")
    print(f"  Recall:    {orig_best['recall']*100:.2f}%")
    print(f"  F1-Score:  {orig_best['f1_score']*100:.2f}%")
    print(f"  ROC AUC:   {orig_best['roc_auc']*100:.2f}%")
    print(f"  TP: {orig_best['tp']}, FP: {orig_best['fp']}")
    
    print(f"\nBaseline Model (Best F1 at threshold {base_best['threshold']:.2f}):")
    print(f"  Precision: {base_best['precision']*100:.2f}%")
    print(f"  Recall:    {base_best['recall']*100:.2f}%")
    print(f"  F1-Score:  {base_best['f1_score']*100:.2f}%")
    print(f"  ROC AUC:   {base_best['roc_auc']*100:.2f}%" if base_best['roc_auc'] else "  ROC AUC:   N/A")
    print(f"  TP: {base_best['tp']}, FP: {base_best['fp']}")
    
    print(f"\nAutoencoder Model (Best F1 at threshold {auto_best['threshold']:.2f}):")
    print(f"  Precision: {auto_best['precision']*100:.2f}%")
    print(f"  Recall:    {auto_best['recall']*100:.2f}%")
    print(f"  F1-Score:  {auto_best['f1_score']*100:.2f}%")
    print(f"  ROC AUC:   {auto_best['roc_auc']*100:.2f}%" if auto_best['roc_auc'] else "  ROC AUC:   N/A")
    print(f"  TP: {auto_best['tp']}, FP: {auto_best['fp']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"fair_threshold_comparison_{timestamp}.json")
    
    results = {
        'timestamp': timestamp,
        'original_model': {
            'thresholds': ORIGINAL_THRESHOLDS,
            'results': original_results,
            'best': orig_best
        },
        'comprehensive_baseline': {
            'thresholds': COMPREHENSIVE_THRESHOLDS,
            'results': baseline_results,
            'best': base_best
        },
        'comprehensive_autoencoder': {
            'thresholds': COMPREHENSIVE_THRESHOLDS,
            'results': autoencoder_results,
            'best': auto_best
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("FAIR COMPARISON COMPLETED")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())

