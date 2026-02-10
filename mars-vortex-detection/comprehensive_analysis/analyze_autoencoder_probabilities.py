#!/usr/bin/env python3
"""
Deep Dive: Why Autoencoder Model Yields Zeros at Threshold 0.03+
================================================================

This script analyzes the probability distribution of the autoencoder model
to understand why it produces no positive predictions at thresholds >= 0.03.
"""

import os
import pandas as pd
import numpy as np
import glob
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# =============================================================================
# LOAD DATA
# =============================================================================

def load_models_and_data():
    """Load autoencoder model and test data."""
    print("=" * 70)
    print("LOADING AUTOENCODER MODEL AND TEST DATA")
    print("=" * 70)
    
    # Load autoencoder model
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    if not autoencoder_files:
        print("[ERROR] No autoencoder model found!")
        return None, None, None
    
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    print(f"Loaded autoencoder model: {max(autoencoder_files, key=os.path.getctime)}")
    
    # Load test data
    features_df = pd.read_csv(os.path.join(FEATURES_DIR, "test_sliding_features_step10.csv"))
    print(f"Loaded {len(features_df):,} test samples")
    
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    X = features_df[feature_cols].values
    y = features_df['label'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Positive samples: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    print(f"Negative samples: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")
    
    return autoencoder_model, X, y, feature_cols

# =============================================================================
# ANALYZE PROBABILITIES
# =============================================================================

def analyze_probability_distribution(model, X, y):
    """Analyze probability distribution in detail."""
    print("\n" + "=" * 70)
    print("PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]
    
    # Overall statistics
    print("\n📊 Overall Statistics:")
    print(f"  Min:     {y_proba.min():.6f} ({y_proba.min()*100:.4f}%)")
    print(f"  Max:     {y_proba.max():.6f} ({y_proba.max()*100:.4f}%)")
    print(f"  Mean:    {y_proba.mean():.6f} ({y_proba.mean()*100:.4f}%)")
    print(f"  Median:  {np.median(y_proba):.6f} ({np.median(y_proba)*100:.4f}%)")
    print(f"  Std:     {y_proba.std():.6f}")
    
    # Percentiles
    print("\n📈 Percentiles:")
    for p in [50, 75, 90, 95, 99, 99.9, 99.99]:
        val = np.percentile(y_proba, p)
        print(f"  {p:5.2f}th: {val:.6f} ({val*100:.4f}%)")
    
    # Count samples above thresholds
    print("\n🔍 Samples Above Thresholds:")
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10]
    for thresh in thresholds:
        count = (y_proba >= thresh).sum()
        pct = count / len(y_proba) * 100
        print(f"  >= {thresh:.2f}: {count:6d} samples ({pct:.2f}%)")
    
    # Positive vs Negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    print("\n✅ Positive Samples (True Vortices):")
    if len(pos_indices) > 0:
        pos_proba = y_proba[pos_indices]
        print(f"  Count:   {len(pos_indices)}")
        print(f"  Min:     {pos_proba.min():.6f} ({pos_proba.min()*100:.4f}%)")
        print(f"  Max:     {pos_proba.max():.6f} ({pos_proba.max()*100:.4f}%)")
        print(f"  Mean:    {pos_proba.mean():.6f} ({pos_proba.mean()*100:.4f}%)")
        print(f"  Median:  {np.median(pos_proba):.6f} ({np.median(pos_proba)*100:.4f}%)")
        
        # Count positive samples above thresholds
        print(f"\n  Positive samples above thresholds:")
        for thresh in thresholds:
            count = (pos_proba >= thresh).sum()
            pct = count / len(pos_indices) * 100
            print(f"    >= {thresh:.2f}: {count:3d} samples ({pct:.2f}%)")
    else:
        print("  No positive samples!")
    
    print("\n❌ Negative Samples (Non-Vortices):")
    if len(neg_indices) > 0:
        neg_proba = y_proba[neg_indices]
        print(f"  Count:   {len(neg_indices)}")
        print(f"  Min:     {neg_proba.min():.6f} ({neg_proba.min()*100:.4f}%)")
        print(f"  Max:     {neg_proba.max():.6f} ({neg_proba.max()*100:.4f}%)")
        print(f"  Mean:    {neg_proba.mean():.6f} ({neg_proba.mean()*100:.4f}%)")
        print(f"  Median:  {np.median(neg_proba):.6f} ({np.median(neg_proba)*100:.4f}%)")
        
        # Count negative samples above thresholds
        print(f"\n  Negative samples above thresholds:")
        for thresh in thresholds:
            count = (neg_proba >= thresh).sum()
            pct = count / len(neg_indices) * 100
            print(f"    >= {thresh:.2f}: {count:6d} samples ({pct:.2f}%)")
    else:
        print("  No negative samples!")
    
    # Why zeros at 0.03+?
    print("\n" + "=" * 70)
    print("🔬 WHY ZEROS AT THRESHOLD 0.03+?")
    print("=" * 70)
    
    max_proba = y_proba.max()
    print(f"\nMaximum probability in entire dataset: {max_proba:.6f} ({max_proba*100:.4f}%)")
    
    if max_proba < 0.03:
        print(f"\n❌ PROBLEM IDENTIFIED:")
        print(f"   The maximum probability ({max_proba*100:.4f}%) is LESS than 0.03 (3%)")
        print(f"   Therefore, NO samples can exceed threshold 0.03")
        print(f"   → All predictions will be negative at threshold >= 0.03")
    else:
        count_above_03 = (y_proba >= 0.03).sum()
        print(f"\n✅ {count_above_03} samples have probability >= 0.03")
        print(f"   But they might all be false positives")
    
    # Check separation
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        pos_proba = y_proba[pos_indices]
        neg_proba = y_proba[neg_indices]
        
        print(f"\n📊 Class Separation Analysis:")
        print(f"   Positive mean: {pos_proba.mean():.6f}")
        print(f"   Negative mean: {neg_proba.mean():.6f}")
        print(f"   Difference:    {pos_proba.mean() - neg_proba.mean():.6f}")
        
        if pos_proba.mean() > neg_proba.mean():
            print(f"   ✅ Model CAN distinguish (positive samples have higher probabilities)")
        else:
            print(f"   ❌ Model CANNOT distinguish (negative samples have higher probabilities)")
    
    return y_proba

# =============================================================================
# VISUALIZE
# =============================================================================

def create_detailed_visualization(y_proba, y):
    """Create detailed visualization of probability distribution."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATION")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Autoencoder Model: Why Zeros at Threshold 0.03+?', 
                fontsize=16, fontweight='bold')
    
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # 1. Histogram - All samples
    ax1 = axes[0, 0]
    ax1.hist(y_proba, bins=100, alpha=0.7, color='gray', edgecolor='black', density=True)
    ax1.axvline(x=0.01, color='green', linestyle='--', linewidth=2, label='Threshold 0.01')
    ax1.axvline(x=0.02, color='orange', linestyle='--', linewidth=2, label='Threshold 0.02')
    ax1.axvline(x=0.03, color='red', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax1.axvline(x=y_proba.max(), color='purple', linestyle=':', linewidth=2, label=f'Max Prob ({y_proba.max():.4f})')
    ax1.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Probability Distribution (All Samples)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(0.05, y_proba.max() * 1.1))
    
    # 2. Histogram - Positive vs Negative
    ax2 = axes[0, 1]
    if len(pos_indices) > 0:
        ax2.hist(y_proba[pos_indices], bins=50, alpha=0.7, label='Positive (Vortices)', 
                color='green', edgecolor='black', density=True)
    if len(neg_indices) > 0:
        ax2.hist(y_proba[neg_indices], bins=100, alpha=0.5, label='Negative (Non-Vortices)', 
                color='red', edgecolor='black', density=True)
    ax2.axvline(x=0.03, color='black', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Distribution (Positive vs Negative)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(0.05, y_proba.max() * 1.1))
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_proba = np.sort(y_proba)
    ax3.plot(sorted_proba, np.arange(len(sorted_proba))/len(sorted_proba), 
            linewidth=2, color='blue')
    ax3.axvline(x=0.01, color='green', linestyle='--', linewidth=2, label='Threshold 0.01')
    ax3.axvline(x=0.02, color='orange', linestyle='--', linewidth=2, label='Threshold 0.02')
    ax3.axvline(x=0.03, color='red', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax3.axvline(x=y_proba.max(), color='purple', linestyle=':', linewidth=2, label=f'Max Prob')
    ax3.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Fraction', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(0.05, y_proba.max() * 1.1))
    
    # 4. Predictions at different thresholds
    ax4 = axes[1, 1]
    thresholds = np.arange(0.005, 0.035, 0.001)
    predictions = [(y_proba >= t).sum() for t in thresholds]
    
    ax4.plot(thresholds, predictions, linewidth=2, color='blue', marker='o', markersize=3)
    ax4.axvline(x=0.01, color='green', linestyle='--', linewidth=2, label='Threshold 0.01')
    ax4.axvline(x=0.02, color='orange', linestyle='--', linewidth=2, label='Threshold 0.02')
    ax4.axvline(x=0.03, color='red', linestyle='--', linewidth=2, label='Threshold 0.03')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Number of Positive Predictions', fontsize=12, fontweight='bold')
    ax4.set_title('Predictions vs Threshold', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.005, 0.035)
    
    output_file = os.path.join(RESULTS_DIR, f"autoencoder_probability_analysis_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved visualization to: {output_file}")
    plt.close()
    
    return output_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("AUTOENCODER MODEL: PROBABILITY DISTRIBUTION DEEP DIVE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    model, X, y, feature_cols = load_models_and_data()
    if model is None:
        return 1
    
    # Analyze probabilities
    y_proba = analyze_probability_distribution(model, X, y)
    
    # Create visualization
    viz_file = create_detailed_visualization(y_proba, y)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nVisualization saved to: {viz_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())





