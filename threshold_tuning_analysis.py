#!/usr/bin/env python3
"""
Threshold Tuning and Analysis
==============================

This script performs comprehensive threshold analysis to find optimal
decision thresholds for the retrained Random Forest model.

Steps:
1. Load retrained model and features
2. Evaluate on validation set across multiple thresholds
3. Find optimal threshold (maximize F1-score)
4. Apply optimal threshold to test set
5. Compare results and generate report
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# Find latest model
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Thresholds to evaluate
THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

def find_latest_model():
    """Find the most recently trained model."""
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("rf_vortex_detector_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No trained model found in models/ directory")
    
    # Sort by timestamp (filename format: rf_vortex_detector_YYYYMMDD_HHMMSS.pkl)
    model_files.sort(reverse=True)
    latest_model = os.path.join(MODELS_DIR, model_files[0])
    
    print(f"Loading latest model: {model_files[0]}")
    return latest_model

def load_model_and_data():
    """Load model and feature datasets."""
    print("=" * 70)
    print("LOADING MODEL AND DATA")
    print("=" * 70)
    
    # Load model
    model_path = find_latest_model()
    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")
    
    # Load features
    train_df = pd.read_csv("datasets/train_features.csv")
    val_df = pd.read_csv("datasets/val_features.csv")
    test_df = pd.read_csv("datasets/test_features.csv")
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Prepare features
    exclude_cols = ['window_id', 'event_sclk', 'label']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Feature names: {feature_cols}")
    
    return model, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

# =============================================================================
# THRESHOLD EVALUATION
# =============================================================================

def evaluate_at_threshold(model, X, y, threshold, split_name=""):
    """Evaluate model at a specific threshold."""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC AUC (doesn't depend on threshold)
    roc_auc = roc_auc_score(y, y_proba)
    
    return {
        'threshold': threshold,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'predictions': int((y_pred == 1).sum())
    }

def evaluate_multiple_thresholds(model, X, y, thresholds, split_name=""):
    """Evaluate model across multiple thresholds."""
    print(f"\nEvaluating {split_name} set across {len(thresholds)} thresholds...")
    
    results = []
    for threshold in thresholds:
        result = evaluate_at_threshold(model, X, y, threshold, split_name)
        results.append(result)
    
    return results

def find_optimal_threshold(results, metric='f1_score'):
    """Find threshold that maximizes specified metric."""
    best_idx = np.argmax([r[metric] for r in results])
    return results[best_idx]

# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def print_threshold_table(results, split_name=""):
    """Print formatted threshold results table."""
    print(f"\n{'=' * 70}")
    print(f"THRESHOLD ANALYSIS - {split_name.upper()} SET")
    print(f"{'=' * 70}")
    
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['threshold']:<12.2f} {r['precision']:<12.4f} {r['recall']:<12.4f} "
              f"{r['f1_score']:<12.4f} {r['tp']:<8} {r['fp']:<8} {r['fn']:<8}")
    
    # Find best
    best_f1 = find_optimal_threshold(results, 'f1_score')
    best_precision = find_optimal_threshold(results, 'precision')
    best_recall = find_optimal_threshold(results, 'recall')
    
    print(f"\n{'=' * 70}")
    print(f"OPTIMAL THRESHOLDS - {split_name.upper()} SET")
    print(f"{'=' * 70}")
    print(f"Best F1-Score:  threshold={best_f1['threshold']:.2f}, "
          f"F1={best_f1['f1_score']:.4f}, "
          f"Precision={best_f1['precision']:.4f}, "
          f"Recall={best_f1['recall']:.4f}")
    print(f"Best Precision: threshold={best_precision['threshold']:.2f}, "
          f"Precision={best_precision['precision']:.4f}, "
          f"F1={best_precision['f1_score']:.4f}")
    print(f"Best Recall:    threshold={best_recall['threshold']:.2f}, "
          f"Recall={best_recall['recall']:.4f}, "
          f"F1={best_recall['f1_score']:.4f}")

def compare_validation_to_test(val_results, test_results):
    """Compare validation and test performance."""
    print(f"\n{'=' * 70}")
    print("VALIDATION vs TEST COMPARISON")
    print(f"{'=' * 70}")
    
    # Find best validation threshold
    best_val = find_optimal_threshold(val_results, 'f1_score')
    val_threshold = best_val['threshold']
    
    print(f"\nUsing validation-optimal threshold: {val_threshold:.2f}")
    print(f"  Validation F1: {best_val['f1_score']:.4f}")
    
    # Find test performance at validation threshold
    test_at_val_threshold = next(
        (r for r in test_results if abs(r['threshold'] - val_threshold) < 0.01),
        None
    )
    
    if test_at_val_threshold:
        print(f"  Test F1:      {test_at_val_threshold['f1_score']:.4f}")
        print(f"  Test Precision: {test_at_val_threshold['precision']:.4f}")
        print(f"  Test Recall:    {test_at_val_threshold['recall']:.4f}")
        
        print(f"\n{'Metric':<15} {'Validation':<15} {'Test':<15} {'Difference':<15}")
        print("-" * 60)
        print(f"{'F1-Score':<15} {best_val['f1_score']:<15.4f} {test_at_val_threshold['f1_score']:<15.4f} "
              f"{test_at_val_threshold['f1_score'] - best_val['f1_score']:<15.4f}")
        print(f"{'Precision':<15} {best_val['precision']:<15.4f} {test_at_val_threshold['precision']:<15.4f} "
              f"{test_at_val_threshold['precision'] - best_val['precision']:<15.4f}")
        print(f"{'Recall':<15} {best_val['recall']:<15.4f} {test_at_val_threshold['recall']:<15.4f} "
              f"{test_at_val_threshold['recall'] - best_val['recall']:<15.4f}")
    
    # Find best test threshold
    best_test = find_optimal_threshold(test_results, 'f1_score')
    print(f"\nTest-optimal threshold: {best_test['threshold']:.2f}")
    print(f"  Test F1:      {best_test['f1_score']:.4f}")
    print(f"  Test Precision: {best_test['precision']:.4f}")
    print(f"  Test Recall:    {best_test['recall']:.4f}")

def plot_threshold_curves(val_results, test_results):
    """Plot precision, recall, and F1 curves across thresholds."""
    print(f"\nGenerating threshold analysis plots...")
    
    thresholds = [r['threshold'] for r in val_results]
    val_precision = [r['precision'] for r in val_results]
    val_recall = [r['recall'] for r in val_results]
    val_f1 = [r['f1_score'] for r in val_results]
    
    test_precision = [r['precision'] for r in test_results]
    test_recall = [r['recall'] for r in test_results]
    test_f1 = [r['f1_score'] for r in test_results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Precision
    axes[0].plot(thresholds, val_precision, 'o-', label='Validation', linewidth=2)
    axes[0].plot(thresholds, test_precision, 's-', label='Test', linewidth=2)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].plot(thresholds, val_recall, 'o-', label='Validation', linewidth=2)
    axes[1].plot(thresholds, test_recall, 's-', label='Test', linewidth=2)
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall vs Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[2].plot(thresholds, val_f1, 'o-', label='Validation', linewidth=2)
    axes[2].plot(thresholds, test_f1, 's-', label='Test', linewidth=2)
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_title('F1-Score vs Threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(RESULTS_DIR, f"threshold_analysis_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"  Plot saved: {plot_file}")
    
    plt.close()

def save_results(val_results, test_results):
    """Save threshold analysis results to JSON."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find optimal thresholds
    best_val = find_optimal_threshold(val_results, 'f1_score')
    best_test = find_optimal_threshold(test_results, 'f1_score')
    
    results = {
        'timestamp': timestamp,
        'validation_results': val_results,
        'test_results': test_results,
        'optimal_thresholds': {
            'validation': {
                'threshold': best_val['threshold'],
                'precision': best_val['precision'],
                'recall': best_val['recall'],
                'f1_score': best_val['f1_score']
            },
            'test': {
                'threshold': best_test['threshold'],
                'precision': best_test['precision'],
                'recall': best_test['recall'],
                'f1_score': best_test['f1_score']
            }
        }
    }
    
    results_file = os.path.join(RESULTS_DIR, f"threshold_analysis_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved: {results_file}")
    return results_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main threshold tuning pipeline."""
    print("=" * 70)
    print("THRESHOLD TUNING AND ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Evaluating {len(THRESHOLDS)} thresholds: {THRESHOLDS}")
    print("=" * 70)
    
    # Load model and data
    model, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_model_and_data()
    
    # Evaluate on validation set
    val_results = evaluate_multiple_thresholds(model, X_val, y_val, THRESHOLDS, "Validation")
    print_threshold_table(val_results, "Validation")
    
    # Evaluate on test set
    test_results = evaluate_multiple_thresholds(model, X_test, y_test, THRESHOLDS, "Test")
    print_threshold_table(test_results, "Test")
    
    # Compare validation and test
    compare_validation_to_test(val_results, test_results)
    
    # Generate plots
    plot_threshold_curves(val_results, test_results)
    
    # Save results
    results_file = save_results(val_results, test_results)
    
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"\nSummary:")
    best_val = find_optimal_threshold(val_results, 'f1_score')
    best_test = find_optimal_threshold(test_results, 'f1_score')
    print(f"  Validation optimal: threshold={best_val['threshold']:.2f}, F1={best_val['f1_score']:.4f}")
    print(f"  Test optimal:       threshold={best_test['threshold']:.2f}, F1={best_test['f1_score']:.4f}")
    print(f"\nResults saved to: {results_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
