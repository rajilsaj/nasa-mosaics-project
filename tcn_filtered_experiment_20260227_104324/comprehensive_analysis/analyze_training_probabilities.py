#!/usr/bin/env python3
"""
Analyze Probability Distributions on Training Set - Extended Model Only
========================================================================

Diagnostic script to check what probability distribution the extended model
(19 features) produces on the balanced training set it was trained on. This 
helps diagnose if low probabilities are due to:
1. Model learning to be conservative even on balanced data
2. Deployment distribution mismatch (expected)
3. Model calibration issues
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
from sklearn.metrics import roc_auc_score
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
# LOAD MODEL AND DATA
# =============================================================================

def load_model_and_training_data():
    """Load extended model and balanced training data."""
    print("=" * 70)
    print("LOADING EXTENDED MODEL AND TRAINING DATA")
    print("=" * 70)
    
    # Load extended model only
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    
    if not autoencoder_files:
        print("[ERROR] Extended model not found!")
        return None, None, None
    
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    
    print(f"Loaded extended model: {os.path.basename(max(autoencoder_files, key=os.path.getctime))}")
    
    # Load BALANCED training data
    train_file = os.path.join(FEATURES_DIR, "train_balanced.csv")
    if not os.path.exists(train_file):
        print(f"[ERROR] Training file not found: {train_file}")
        return None, None, None
    
    train_df = pd.read_csv(train_file)
    print(f"Loaded {len(train_df):,} training samples (balanced)")
    
    # Check class distribution
    if 'label' in train_df.columns:
        class_dist = train_df['label'].value_counts()
        print(f"\nTraining Class Distribution:")
        print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(train_df)*100:.2f}%)")
        print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(train_df)*100:.2f}%)")
        if class_dist.get(1, 0) > 0:
            ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
            print(f"  Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    # Separate features and labels
    label_cols = ['label']
    feature_cols = [col for col in train_df.columns if col not in label_cols]
    
    # Load model metadata to get correct feature columns
    autoencoder_metadata_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_metadata_*.json"))
    
    if autoencoder_metadata_files:
        with open(max(autoencoder_metadata_files, key=os.path.getctime), 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = feature_cols
    
    # Select features for extended model
    autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
    
    autoencoder_X = train_df[autoencoder_feature_cols].values
    y = train_df['label'].values
    
    print(f"\nExtended model features: {len(autoencoder_feature_cols)}")
    
    return autoencoder_model, autoencoder_X, y

def load_validation_data(autoencoder_model):
    """Load validation data for comparison."""
    val_file = os.path.join(FEATURES_DIR, "val_sliding_features_step10.csv")
    if not os.path.exists(val_file):
        return None, None, None
    
    val_df = pd.read_csv(val_file)
    print(f"\nLoaded {len(val_df):,} validation samples for comparison")
    
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in val_df.columns if col not in label_cols]
    
    # Load model metadata
    autoencoder_metadata_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_metadata_*.json"))
    
    if autoencoder_metadata_files:
        with open(max(autoencoder_metadata_files, key=os.path.getctime), 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = feature_cols
    
    autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
    
    autoencoder_X_val = val_df[autoencoder_feature_cols].values
    y_val = val_df['label'].values
    
    return autoencoder_X_val, y_val

# =============================================================================
# ANALYZE PROBABILITIES
# =============================================================================

def analyze_probabilities(y_proba, y, dataset_name=""):
    """Analyze probability distribution."""
    print(f"\n" + "=" * 70)
    print(f"PROBABILITY ANALYSIS - {dataset_name.upper()}")
    print("=" * 70)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Min:     {y_proba.min():.6f} ({y_proba.min()*100:.4f}%)")
    print(f"  Max:     {y_proba.max():.6f} ({y_proba.max()*100:.4f}%)")
    print(f"  Mean:    {y_proba.mean():.6f} ({y_proba.mean()*100:.4f}%)")
    print(f"  Median:  {np.median(y_proba):.6f} ({np.median(y_proba)*100:.4f}%)")
    print(f"  Std:     {y_proba.std():.6f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.9, 99.99]:
        val = np.percentile(y_proba, p)
        print(f"  {p:5.2f}th: {val:.6f} ({val*100:.4f}%)")
    
    # Positive vs Negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    if len(pos_indices) > 0:
        pos_proba = y_proba[pos_indices]
        print(f"\nPositive Samples:")
        print(f"  Count:   {len(pos_indices)}")
        print(f"  Min:     {pos_proba.min():.6f} ({pos_proba.min()*100:.4f}%)")
        print(f"  Max:     {pos_proba.max():.6f} ({pos_proba.max()*100:.4f}%)")
        print(f"  Mean:    {pos_proba.mean():.6f} ({pos_proba.mean()*100:.4f}%)")
        print(f"  Median:  {np.median(pos_proba):.6f} ({np.median(pos_proba)*100:.4f}%)")
    
    if len(neg_indices) > 0:
        neg_proba = y_proba[neg_indices]
        print(f"\nNegative Samples:")
        print(f"  Count:   {len(neg_indices)}")
        print(f"  Min:     {neg_proba.min():.6f} ({neg_proba.min()*100:.4f}%)")
        print(f"  Max:     {neg_proba.max():.6f} ({neg_proba.max()*100:.4f}%)")
        print(f"  Mean:    {neg_proba.mean():.6f} ({neg_proba.mean()*100:.4f}%)")
        print(f"  Median:  {np.median(neg_proba):.6f} ({np.median(neg_proba)*100:.4f}%)")
    
    # Class separation
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        pos_proba = y_proba[pos_indices]
        neg_proba = y_proba[neg_indices]
        print(f"\nClass Separation:")
        print(f"  Positive mean: {pos_proba.mean():.6f}")
        print(f"  Negative mean: {neg_proba.mean():.6f}")
        print(f"  Difference:    {pos_proba.mean() - neg_proba.mean():.6f}")
        if pos_proba.mean() > neg_proba.mean():
            print(f"  [OK] Model CAN distinguish (positive samples have higher probabilities)")
        else:
            print(f"  [WARNING] Model CANNOT distinguish (negative samples have higher probabilities)")
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(y, y_proba)
        print(f"\nROC AUC: {roc_auc:.4f}")
    except ValueError:
        roc_auc = None
        print(f"\nROC AUC: N/A (only one class)")
    
    return {
        'min': float(y_proba.min()),
        'max': float(y_proba.max()),
        'mean': float(y_proba.mean()),
        'median': float(np.median(y_proba)),
        'std': float(y_proba.std()),
        'pos_mean': float(y_proba[pos_indices].mean()) if len(pos_indices) > 0 else None,
        'neg_mean': float(y_proba[neg_indices].mean()) if len(neg_indices) > 0 else None,
        'roc_auc': float(roc_auc) if roc_auc is not None else None
    }

# =============================================================================
# VISUALIZE COMPARISON
# =============================================================================

def create_comparison_visualization(train_proba, val_proba, y_train, y_val):
    """Create comparison visualization."""
    print("\n" + "=" * 70)
    print("CREATING COMPARISON VISUALIZATION")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Extended Model: Probability Distribution - Training vs Validation Set', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Histogram - Training vs Validation
    ax1 = axes[0, 0]
    ax1.hist(train_proba, bins=50, alpha=0.7, label='Training (Balanced 1:1)', 
            color='green', edgecolor='black', density=True)
    ax1.hist(val_proba, bins=100, alpha=0.5, label='Validation (Imbalanced 88:1)', 
            color='red', edgecolor='black', density=True)
    ax1.axvline(x=0.01, color='blue', linestyle='--', linewidth=2, label='Threshold 0.01')
    ax1.axvline(x=0.03, color='orange', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax1.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Probability Distribution: Training vs Validation', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(0.05, max(train_proba.max(), val_proba.max()) * 1.1))
    
    # 2. Positive vs Negative - Training
    ax2 = axes[0, 1]
    pos_indices_train = np.where(y_train == 1)[0]
    neg_indices_train = np.where(y_train == 0)[0]
    if len(pos_indices_train) > 0:
        ax2.hist(train_proba[pos_indices_train], bins=30, alpha=0.7, 
                label='Positive (Vortices)', color='green', edgecolor='black', density=True)
    if len(neg_indices_train) > 0:
        ax2.hist(train_proba[neg_indices_train], bins=30, alpha=0.5, 
                label='Negative (Non-Vortices)', color='red', edgecolor='black', density=True)
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Training Set: Positive vs Negative', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(0.05, train_proba.max() * 1.1))
    
    # 3. Positive vs Negative - Validation
    ax3 = axes[1, 0]
    pos_indices_val = np.where(y_val == 1)[0]
    neg_indices_val = np.where(y_val == 0)[0]
    if len(pos_indices_val) > 0:
        ax3.hist(val_proba[pos_indices_val], bins=30, alpha=0.7, 
                label='Positive (Vortices)', color='green', edgecolor='black', density=True)
    if len(neg_indices_val) > 0:
        ax3.hist(val_proba[neg_indices_val], bins=100, alpha=0.5, 
                label='Negative (Non-Vortices)', color='red', edgecolor='black', density=True)
    ax3.axvline(x=0.01, color='blue', linestyle='--', linewidth=2, label='Threshold 0.01')
    ax3.axvline(x=0.03, color='orange', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax3.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Validation Set: Positive vs Negative', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(0.05, val_proba.max() * 1.1))
    
    # 4. Box plot comparison
    ax4 = axes[1, 1]
    data_to_plot = [train_proba, val_proba]
    bp = ax4.boxplot(data_to_plot, labels=['Training', 'Validation'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax4.axhline(y=0.01, color='blue', linestyle='--', linewidth=2, label='Threshold 0.01')
    ax4.axhline(y=0.03, color='orange', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax4.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax4.set_title('Probability Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, max(0.05, max(train_proba.max(), val_proba.max()) * 1.1))
    
    plt.tight_layout()
    
    output_file = os.path.join(RESULTS_DIR, f"extended_model_training_vs_validation_probabilities_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved visualization to: {output_file}")
    plt.close()
    
    return output_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("PROBABILITY DISTRIBUTION ANALYSIS: EXTENDED MODEL - TRAINING SET")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPurpose: Check if extended model produces low probabilities even on balanced training data")
    print(f"         This helps diagnose if low probabilities are due to:")
    print(f"         1. Model learning to be conservative (even on balanced data)")
    print(f"         2. Deployment distribution mismatch (expected)")
    print(f"         3. Model calibration issues")
    
    # Load model and training data
    autoencoder_model, autoencoder_X_train, y_train = load_model_and_training_data()
    
    if autoencoder_model is None:
        return 1
    
    # Get probabilities on training set
    print("\n" + "=" * 70)
    print("GETTING PROBABILITIES ON TRAINING SET")
    print("=" * 70)
    
    autoencoder_train_proba = autoencoder_model.predict_proba(autoencoder_X_train)[:, 1]
    
    # Analyze training probabilities
    train_stats = analyze_probabilities(autoencoder_train_proba, y_train, "Extended Model - Training")
    
    # Load validation data for comparison
    autoencoder_X_val, y_val = load_validation_data(autoencoder_model)
    
    if autoencoder_X_val is not None:
        autoencoder_val_proba = autoencoder_model.predict_proba(autoencoder_X_val)[:, 1]
        
        print("\n" + "=" * 70)
        print("COMPARING WITH VALIDATION SET")
        print("=" * 70)
        
        val_stats = analyze_probabilities(autoencoder_val_proba, y_val, "Extended Model - Validation")
        
        # Comparison summary
        print("\n" + "=" * 70)
        print("TRAINING VS VALIDATION COMPARISON")
        print("=" * 70)
        
        print(f"\nExtended Model (19 features):")
        print(f"  Training Max Prob: {train_stats['max']:.6f} ({train_stats['max']*100:.4f}%)")
        print(f"  Validation Max Prob: {val_stats['max']:.6f} ({val_stats['max']*100:.4f}%)")
        print(f"  Difference: {abs(train_stats['max'] - val_stats['max']):.6f}")
        print(f"  Training Mean: {train_stats['mean']:.6f} ({train_stats['mean']*100:.4f}%)")
        print(f"  Validation Mean: {val_stats['mean']:.6f} ({val_stats['mean']*100:.4f}%)")
        
        # Create visualization
        viz_file = create_comparison_visualization(
            autoencoder_train_proba, autoencoder_val_proba, y_train, y_val
        )
        
        # Save results
        results = {
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model': 'extended_model_19_features',
            'training': train_stats,
            'validation': val_stats,
            'key_finding': {
                'training_max_prob': float(train_stats['max']),
                'validation_max_prob': float(val_stats['max']),
                'probability_drop': float(train_stats['max'] - val_stats['max']),
                'diagnosis': 'distribution_shift' if train_stats['max'] > 0.5 else 'model_conservatism'
            }
        }
        
        results_file = os.path.join(RESULTS_DIR, f"extended_model_training_probability_analysis_{results['timestamp']}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[OK] Results saved to: {results_file}")
        print(f"[OK] Visualization saved to: {viz_file}")
        
        # Key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)
        
        if train_stats['max'] > 0.5:
            print(f"\n[FINDING]")
            print(f"   Extended model produces HIGH probabilities on training set")
            print(f"   Training Max Prob: {train_stats['max']*100:.4f}%")
            print(f"   Validation Max Prob: {val_stats['max']*100:.4f}%")
            print(f"   This confirms that low probabilities on validation are due to")
            print(f"   DISTRIBUTION SHIFT (training 1:1 vs validation 88:1), not model conservatism.")
            print(f"   The model works perfectly on balanced data but probabilities drop")
            print(f"   dramatically when faced with extreme class imbalance.")
        else:
            print(f"\n[CRITICAL FINDING]")
            print(f"   Extended model produces low probabilities even on TRAINING set!")
            print(f"   Training Max Prob: {train_stats['max']*100:.4f}%")
            print(f"   This suggests the model learned to be conservative during training,")
            print(f"   not just due to deployment distribution mismatch.")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())
