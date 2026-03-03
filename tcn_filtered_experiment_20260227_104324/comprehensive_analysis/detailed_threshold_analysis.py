#!/usr/bin/env python3
"""
Detailed Threshold Analysis for Sliding Windows
================================================

As a seasoned RF scientist, this script:
1. Tests multiple thresholds (0.01 to 0.30)
2. Generates precision-recall and ROC curves
3. Finds optimal thresholds for different metrics
4. Compares baseline vs autoencoder models
5. Provides deployment-ready recommendations
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import joblib
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
        description='Detailed threshold analysis for sliding windows',
        epilog="""
Examples:
  python detailed_threshold_analysis.py --split val --step_size 10
  python detailed_threshold_analysis.py --split test --step_size 10
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
    
    return parser.parse_args()

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
    
    # Get predictions (baseline uses first 15 features, autoencoder uses all)
    baseline_X = X[:, :15]
    autoencoder_X = X
    
    return baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, feature_cols

# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================

def analyze_thresholds(model, X, y, model_name=""):
    """Analyze model performance across multiple thresholds."""
    print(f"\nAnalyzing thresholds for {model_name}...")
    
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Test multiple thresholds
    thresholds = np.arange(0.01, 0.31, 0.01)
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        
        cm = confusion_matrix(y, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tpr': tpr,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    results_df = pd.DataFrame(results)
    
    # Calculate ROC and PR curves
    try:
        fpr_curve, tpr_curve, _ = roc_curve(y, y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
        roc_auc = roc_auc_score(y, y_proba)
        pr_auc = average_precision_score(y, y_proba)
    except ValueError:
        fpr_curve, tpr_curve, _ = np.array([]), np.array([]), np.array([])
        precision_curve, recall_curve, _ = np.array([]), np.array([]), np.array([])
        roc_auc = np.nan
        pr_auc = np.nan
    
    # Find optimal thresholds
    optimal_f1 = results_df.loc[results_df['f1_score'].idxmax()]
    optimal_precision = results_df.loc[results_df['precision'].idxmax()]
    optimal_recall = results_df.loc[results_df['recall'].idxmax()]
    
    # Find threshold with 90% precision (if possible)
    high_precision = results_df[results_df['precision'] >= 0.90]
    if len(high_precision) > 0:
        optimal_90prec = high_precision.loc[high_precision['f1_score'].idxmax()]
    else:
        optimal_90prec = None
    
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"\n  Optimal F1 threshold: {optimal_f1['threshold']:.3f}")
    print(f"    Precision: {optimal_f1['precision']:.4f}, Recall: {optimal_f1['recall']:.4f}, F1: {optimal_f1['f1_score']:.4f}")
    
    return {
        'results_df': results_df,
        'y_proba': y_proba,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'fpr_curve': fpr_curve,
        'tpr_curve': tpr_curve,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve,
        'optimal_f1': optimal_f1,
        'optimal_precision': optimal_precision,
        'optimal_recall': optimal_recall,
        'optimal_90prec': optimal_90prec
    }

# =============================================================================
# CREATE VISUALIZATIONS
# =============================================================================

def create_visualizations(baseline_analysis, autoencoder_analysis, split_name, step_size):
    """Create threshold analysis visualizations."""
    print(f"\nCreating visualizations...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Threshold Analysis - {split_name.upper()} Sliding Windows (Step Size: {step_size})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Precision-Recall Curve
    ax1 = axes[0, 0]
    if len(autoencoder_analysis['precision_curve']) > 0:
        ax1.plot(autoencoder_analysis['recall_curve'], autoencoder_analysis['precision_curve'], 
                label=f"Autoencoder (AUC={autoencoder_analysis['pr_auc']:.3f})", linewidth=2, color='blue')
    if len(baseline_analysis['precision_curve']) > 0:
        ax1.plot(baseline_analysis['recall_curve'], baseline_analysis['precision_curve'], 
                label=f"Baseline (AUC={baseline_analysis['pr_auc']:.3f})", linewidth=2, color='red', linestyle='--')
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC Curve
    ax2 = axes[0, 1]
    if len(autoencoder_analysis['tpr_curve']) > 0:
        ax2.plot(autoencoder_analysis['fpr_curve'], autoencoder_analysis['tpr_curve'], 
                label=f"Autoencoder (AUC={autoencoder_analysis['roc_auc']:.3f})", linewidth=2, color='blue')
    if len(baseline_analysis['tpr_curve']) > 0:
        ax2.plot(baseline_analysis['fpr_curve'], baseline_analysis['tpr_curve'], 
                label=f"Baseline (AUC={baseline_analysis['roc_auc']:.3f})", linewidth=2, color='red', linestyle='--')
    ax2.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1, alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Precision vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(baseline_analysis['results_df']['threshold'], baseline_analysis['results_df']['precision'], 
            label='Baseline', linewidth=2, color='red', linestyle='--')
    ax3.plot(autoencoder_analysis['results_df']['threshold'], autoencoder_analysis['results_df']['precision'], 
            label='Autoencoder', linewidth=2, color='blue')
    ax3.axhline(y=0.90, color='green', linestyle=':', alpha=0.7, label='90% Precision Target')
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision vs Threshold', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Recall vs Threshold
    ax4 = axes[1, 1]
    ax4.plot(baseline_analysis['results_df']['threshold'], baseline_analysis['results_df']['recall'], 
            label='Baseline', linewidth=2, color='red', linestyle='--')
    ax4.plot(autoencoder_analysis['results_df']['threshold'], autoencoder_analysis['results_df']['recall'], 
            label='Autoencoder', linewidth=2, color='blue')
    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Recall vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = os.path.join(RESULTS_DIR, f"{split_name}_threshold_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to: {output_file}")
    
    plt.close()
    
    return output_file

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main threshold analysis pipeline."""
    args = parse_arguments()
    
    print("=" * 70)
    print("DETAILED THRESHOLD ANALYSIS - SLIDING WINDOWS")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Step size: {args.step_size}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models and data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, feature_cols = load_models_and_data(
        args.split, args.step_size
    )
    
    if baseline_model is None:
        return 1
    
    # Analyze thresholds for both models
    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)
    
    baseline_analysis = analyze_thresholds(baseline_model, baseline_X, y, "Baseline Model")
    autoencoder_analysis = analyze_thresholds(autoencoder_model, autoencoder_X, y, "Autoencoder Model")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS COMPARISON")
    print("=" * 70)
    
    print(f"\nBaseline Model (Optimal F1):")
    print(f"  Threshold: {baseline_analysis['optimal_f1']['threshold']:.3f}")
    print(f"  Precision: {baseline_analysis['optimal_f1']['precision']:.4f}")
    print(f"  Recall:    {baseline_analysis['optimal_f1']['recall']:.4f}")
    print(f"  F1-Score:  {baseline_analysis['optimal_f1']['f1_score']:.4f}")
    print(f"  TP: {baseline_analysis['optimal_f1']['tp']}, FP: {baseline_analysis['optimal_f1']['fp']}")
    print(f"  FN: {baseline_analysis['optimal_f1']['fn']}, TN: {baseline_analysis['optimal_f1']['tn']}")
    
    print(f"\nAutoencoder Model (Optimal F1):")
    print(f"  Threshold: {autoencoder_analysis['optimal_f1']['threshold']:.3f}")
    print(f"  Precision: {autoencoder_analysis['optimal_f1']['precision']:.4f}")
    print(f"  Recall:    {autoencoder_analysis['optimal_f1']['recall']:.4f}")
    print(f"  F1-Score:  {autoencoder_analysis['optimal_f1']['f1_score']:.4f}")
    print(f"  TP: {autoencoder_analysis['optimal_f1']['tp']}, FP: {autoencoder_analysis['optimal_f1']['fp']}")
    print(f"  FN: {autoencoder_analysis['optimal_f1']['fn']}, TN: {autoencoder_analysis['optimal_f1']['tn']}")
    
    # Check for 90% precision threshold
    if autoencoder_analysis['optimal_90prec'] is not None:
        print(f"\nAutoencoder Model (90% Precision Target):")
        print(f"  Threshold: {autoencoder_analysis['optimal_90prec']['threshold']:.3f}")
        print(f"  Precision: {autoencoder_analysis['optimal_90prec']['precision']:.4f}")
        print(f"  Recall:    {autoencoder_analysis['optimal_90prec']['recall']:.4f}")
        print(f"  F1-Score:  {autoencoder_analysis['optimal_90prec']['f1_score']:.4f}")
    
    # Create visualizations
    viz_file = create_visualizations(baseline_analysis, autoencoder_analysis, args.split, args.step_size)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"{args.split}_threshold_analysis_{timestamp}.json")
    
    results = {
        'timestamp': timestamp,
        'split': args.split,
        'step_size': args.step_size,
        'baseline': {
            'roc_auc': float(baseline_analysis['roc_auc']) if not np.isnan(baseline_analysis['roc_auc']) else None,
            'pr_auc': float(baseline_analysis['pr_auc']) if not np.isnan(baseline_analysis['pr_auc']) else None,
            'optimal_f1': baseline_analysis['optimal_f1'].to_dict()
        },
        'autoencoder': {
            'roc_auc': float(autoencoder_analysis['roc_auc']) if not np.isnan(autoencoder_analysis['roc_auc']) else None,
            'pr_auc': float(autoencoder_analysis['pr_auc']) if not np.isnan(autoencoder_analysis['pr_auc']) else None,
            'optimal_f1': autoencoder_analysis['optimal_f1'].to_dict(),
            'optimal_90prec': autoencoder_analysis['optimal_90prec'].to_dict() if autoencoder_analysis['optimal_90prec'] is not None else None
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    print("DETAILED THRESHOLD ANALYSIS COMPLETED")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())

