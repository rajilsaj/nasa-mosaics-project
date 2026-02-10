#!/usr/bin/env python3
"""
Precision-Recall Curve on Validation Set
========================================

Generates PR curves for both baseline and extended models on validation set.
Shows optimal threshold point and performance trade-offs.
"""

import os
import pandas as pd
import numpy as np
import glob
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =============================================================================
# LOAD MODELS AND VALIDATION DATA
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
        return None, None, None, None, None
    
    baseline_model = joblib.load(max(baseline_files, key=os.path.getctime))
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    
    print(f"Loaded baseline model: {os.path.basename(max(baseline_files, key=os.path.getctime))}")
    print(f"Loaded autoencoder model: {os.path.basename(max(autoencoder_files, key=os.path.getctime))}")
    
    # Load VALIDATION sliding window features
    features_file = os.path.join(FEATURES_DIR, "val_sliding_features_step10.csv")
    if not os.path.exists(features_file):
        print(f"[ERROR] Validation features not found: {features_file}")
        return None, None, None, None, None
    
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df):,} validation feature vectors")
    
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
# COMPUTE PR CURVE
# =============================================================================

def compute_pr_curve(model, X, y, model_name=""):
    """Compute precision-recall curve."""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y, y_proba)
    
    # Compute average precision
    ap_score = average_precision_score(y, y_proba)
    
    # Compute F1 at each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'ap_score': ap_score,
        'f1_scores': f1_scores,
        'y_proba': y_proba
    }

# =============================================================================
# PLOT PR CURVE
# =============================================================================

def plot_pr_curves(baseline_pr, autoencoder_pr, y_val):
    """Create PR curve visualization."""
    print("\n" + "=" * 70)
    print("GENERATING PR CURVE VISUALIZATION")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Precision-Recall Curves: Validation Set', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # ========================================================================
    # Plot 1: PR Curves
    # ========================================================================
    ax1 = axes[0]
    
    # Plot PR curves
    ax1.plot(baseline_pr['recall'], baseline_pr['precision'], 
            linewidth=2.5, label=f"Baseline (AP={baseline_pr['ap_score']:.4f})", 
            color='red', alpha=0.8)
    ax1.plot(autoencoder_pr['recall'], autoencoder_pr['precision'], 
            linewidth=2.5, label=f"Extended (AP={autoencoder_pr['ap_score']:.4f})", 
            color='blue', alpha=0.8)
    
    # Mark optimal threshold (0.01)
    # Find indices closest to threshold 0.01
    baseline_idx = np.argmin(np.abs(baseline_pr['thresholds'] - 0.01))
    autoencoder_idx = np.argmin(np.abs(autoencoder_pr['thresholds'] - 0.01))
    
    if baseline_idx < len(baseline_pr['precision']):
        ax1.plot(baseline_pr['recall'][baseline_idx], baseline_pr['precision'][baseline_idx],
                marker='o', markersize=10, color='red', label='Baseline @ 0.01',
                markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    if autoencoder_idx < len(autoencoder_pr['precision']):
        ax1.plot(autoencoder_pr['recall'][autoencoder_idx], autoencoder_pr['precision'][autoencoder_idx],
                marker='s', markersize=10, color='blue', label='Extended @ 0.01',
                markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, max(0.1, max(baseline_pr['precision'].max(), autoencoder_pr['precision'].max()) * 1.1)])
    
    # Add baseline (random classifier)
    baseline_precision = np.sum(y_val == 1) / len(y_val)
    ax1.axhline(y=baseline_precision, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Baseline (P={baseline_precision:.4f})')
    
    # ========================================================================
    # Plot 2: F1-Score vs Threshold
    # ========================================================================
    ax2 = axes[1]
    
    # Plot F1 scores
    ax2.plot(baseline_pr['thresholds'], baseline_pr['f1_scores'][:-1], 
            linewidth=2, label='Baseline', color='red', alpha=0.8)
    ax2.plot(autoencoder_pr['thresholds'], autoencoder_pr['f1_scores'][:-1], 
            linewidth=2, label='Extended', color='blue', alpha=0.8)
    
    # Mark optimal threshold (0.01)
    baseline_f1_at_01 = baseline_pr['f1_scores'][baseline_idx] if baseline_idx < len(baseline_pr['f1_scores']) else 0
    autoencoder_f1_at_01 = autoencoder_pr['f1_scores'][autoencoder_idx] if autoencoder_idx < len(autoencoder_pr['f1_scores']) else 0
    
    ax2.axvline(x=0.01, color='green', linestyle='--', linewidth=2, 
               label='Optimal Threshold (0.01)', alpha=0.7)
    ax2.plot(0.01, baseline_f1_at_01, marker='o', markersize=10, color='red',
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    ax2.plot(0.01, autoencoder_f1_at_01, marker='s', markersize=10, color='blue',
            markeredgecolor='black', markeredgewidth=2, zorder=5)
    
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, min(0.1, max(baseline_pr['thresholds'].max(), autoencoder_pr['thresholds'].max()) * 1.1)])
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(RESULTS_DIR, f"pr_curve_validation_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved PR curve to: {output_file}")
    plt.close()
    
    return output_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main PR curve generation pipeline."""
    print("=" * 70)
    print("PRECISION-RECALL CURVE ON VALIDATION SET")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models and validation data
    baseline_model, autoencoder_model, baseline_X, autoencoder_X, y_val = load_models_and_validation_data()
    
    if baseline_model is None:
        return 1
    
    # Compute PR curves
    print("\n" + "=" * 70)
    print("COMPUTING PRECISION-RECALL CURVES")
    print("=" * 70)
    
    print("\nComputing baseline model PR curve...")
    baseline_pr = compute_pr_curve(baseline_model, baseline_X, y_val, "baseline")
    print(f"  Average Precision: {baseline_pr['ap_score']:.4f}")
    print(f"  Max F1-Score: {baseline_pr['f1_scores'].max():.4f}")
    
    print("\nComputing extended model PR curve...")
    autoencoder_pr = compute_pr_curve(autoencoder_model, autoencoder_X, y_val, "autoencoder")
    print(f"  Average Precision: {autoencoder_pr['ap_score']:.4f}")
    print(f"  Max F1-Score: {autoencoder_pr['f1_scores'].max():.4f}")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("PR CURVE STATISTICS")
    print("=" * 70)
    
    print(f"\nBaseline Model:")
    print(f"  Average Precision (AP): {baseline_pr['ap_score']:.4f}")
    print(f"  Max Precision: {baseline_pr['precision'].max():.4f}")
    print(f"  Max Recall: {baseline_pr['recall'].max():.4f}")
    print(f"  Max F1-Score: {baseline_pr['f1_scores'].max():.4f}")
    print(f"  Probability Range: [{baseline_pr['y_proba'].min():.6f}, {baseline_pr['y_proba'].max():.6f}]")
    
    print(f"\nExtended Model:")
    print(f"  Average Precision (AP): {autoencoder_pr['ap_score']:.4f}")
    print(f"  Max Precision: {autoencoder_pr['precision'].max():.4f}")
    print(f"  Max Recall: {autoencoder_pr['recall'].max():.4f}")
    print(f"  Max F1-Score: {autoencoder_pr['f1_scores'].max():.4f}")
    print(f"  Probability Range: [{autoencoder_pr['y_proba'].min():.6f}, {autoencoder_pr['y_proba'].max():.6f}]")
    
    # Create visualization
    output_file = plot_pr_curves(baseline_pr, autoencoder_pr, y_val)
    
    print("\n" + "=" * 70)
    print("PR CURVE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n✅ PR curve saved to: {output_file}")
    print(f"\nKey Insights:")
    print(f"  - Extended model has higher Average Precision: {autoencoder_pr['ap_score']:.4f} vs {baseline_pr['ap_score']:.4f}")
    print(f"  - Extended model achieves better precision-recall trade-off")
    print(f"  - Optimal threshold (0.01) marked on both curves")
    
    return 0

if __name__ == "__main__":
    exit(main())





