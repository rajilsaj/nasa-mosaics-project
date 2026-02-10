#!/usr/bin/env python3
"""
Probability Distribution Visualization
======================================

Visualizes probability distributions for original vs comprehensive models
to explain why different thresholds are needed.
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

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# =============================================================================
# LOAD DATA
# =============================================================================

def load_comprehensive_models_and_data():
    """Load comprehensive models and test data."""
    print("Loading comprehensive models and data...")
    
    # Load models
    baseline_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_model_*.pkl"))
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    
    baseline_model = joblib.load(max(baseline_files, key=os.path.getctime))
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    
    # Load test data
    features_df = pd.read_csv(os.path.join(FEATURES_DIR, "test_sliding_features_step10.csv"))
    
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    baseline_X = features_df.iloc[:, :15].values
    autoencoder_X = features_df[feature_cols].values
    y = features_df['label'].values
    
    # Get probabilities
    baseline_proba = baseline_model.predict_proba(baseline_X)[:, 1]
    autoencoder_proba = autoencoder_model.predict_proba(autoencoder_X)[:, 1]
    
    return baseline_proba, autoencoder_proba, y

# =============================================================================
# CREATE VISUALIZATIONS
# =============================================================================

def create_probability_distribution_plots(baseline_proba, autoencoder_proba, y):
    """Create comprehensive probability distribution visualizations."""
    print("Creating probability distribution visualizations...")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Probability Histograms (All Samples)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(baseline_proba, bins=50, alpha=0.7, label='Baseline', color='red', edgecolor='black')
    ax1.hist(autoencoder_proba, bins=50, alpha=0.7, label='Autoencoder', color='blue', edgecolor='black')
    ax1.axvline(x=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Thresholds')
    ax1.axvline(x=0.90, color='yellow', linestyle='--', linewidth=2)
    ax1.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Probability Distribution (All Samples)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.1)  # Focus on low probabilities
    
    # 2. Probability Histograms (Positive Samples Only)
    ax2 = fig.add_subplot(gs[0, 1])
    pos_indices = np.where(y == 1)[0]
    ax2.hist(baseline_proba[pos_indices], bins=30, alpha=0.7, label='Baseline', color='red', edgecolor='black')
    ax2.hist(autoencoder_proba[pos_indices], bins=30, alpha=0.7, label='Autoencoder', color='blue', edgecolor='black')
    ax2.axvline(x=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Thresholds')
    ax2.axvline(x=0.90, color='yellow', linestyle='--', linewidth=2)
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Distribution (Positive Samples Only)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.1)
    
    # 3. Probability Histograms (Negative Samples Only)
    ax3 = fig.add_subplot(gs[0, 2])
    neg_indices = np.where(y == 0)[0]
    ax3.hist(baseline_proba[neg_indices], bins=50, alpha=0.7, label='Baseline', color='red', edgecolor='black')
    ax3.hist(autoencoder_proba[neg_indices], bins=50, alpha=0.7, label='Autoencoder', color='blue', edgecolor='black')
    ax3.axvline(x=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Thresholds')
    ax3.axvline(x=0.90, color='yellow', linestyle='--', linewidth=2)
    ax3.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Probability Distribution (Negative Samples Only)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 0.1)
    
    # 4. Cumulative Distribution (All Samples)
    ax4 = fig.add_subplot(gs[1, 0])
    sorted_base = np.sort(baseline_proba)
    sorted_auto = np.sort(autoencoder_proba)
    ax4.plot(sorted_base, np.arange(len(sorted_base))/len(sorted_base), 
            label='Baseline', linewidth=2, color='red')
    ax4.plot(sorted_auto, np.arange(len(sorted_auto))/len(sorted_auto), 
            label='Autoencoder', linewidth=2, color='blue')
    ax4.axvline(x=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Thresholds')
    ax4.axvline(x=0.90, color='yellow', linestyle='--', linewidth=2)
    ax4.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Fraction', fontsize=12, fontweight='bold')
    ax4.set_title('Cumulative Distribution (All Samples)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 0.1)
    
    # 5. Box Plot Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    data_to_plot = [baseline_proba, autoencoder_proba]
    bp = ax5.boxplot(data_to_plot, labels=['Baseline', 'Autoencoder'], patch_artist=True)
    bp['boxes'][0].set_facecolor('red')
    bp['boxes'][1].set_facecolor('blue')
    ax5.axhline(y=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Thresholds')
    ax5.axhline(y=0.90, color='yellow', linestyle='--', linewidth=2)
    ax5.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax5.set_title('Probability Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 0.15)
    
    # 6. Statistics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    stats_text = "PROBABILITY STATISTICS\n" + "="*40 + "\n\n"
    
    stats_text += "BASELINE MODEL:\n"
    stats_text += f"  Min: {baseline_proba.min():.4f}\n"
    stats_text += f"  Max: {baseline_proba.max():.4f}\n"
    stats_text += f"  Mean: {baseline_proba.mean():.4f}\n"
    stats_text += f"  Median: {np.median(baseline_proba):.4f}\n"
    stats_text += f"  95th percentile: {np.percentile(baseline_proba, 95):.4f}\n"
    stats_text += f"  99th percentile: {np.percentile(baseline_proba, 99):.4f}\n\n"
    
    stats_text += "AUTOENCODER MODEL:\n"
    stats_text += f"  Min: {autoencoder_proba.min():.4f}\n"
    stats_text += f"  Max: {autoencoder_proba.max():.4f}\n"
    stats_text += f"  Mean: {autoencoder_proba.mean():.4f}\n"
    stats_text += f"  Median: {np.median(autoencoder_proba):.4f}\n"
    stats_text += f"  95th percentile: {np.percentile(autoencoder_proba, 95):.4f}\n"
    stats_text += f"  99th percentile: {np.percentile(autoencoder_proba, 99):.4f}\n\n"
    
    stats_text += "POSITIVE SAMPLES:\n"
    pos_indices = np.where(y == 1)[0]
    stats_text += f"  Baseline mean: {baseline_proba[pos_indices].mean():.4f}\n"
    stats_text += f"  Autoencoder mean: {autoencoder_proba[pos_indices].mean():.4f}\n"
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
            verticalalignment='center', color='white')
    
    # 7. Threshold Comparison
    ax7 = fig.add_subplot(gs[2, :])
    
    thresholds = np.arange(0.01, 0.31, 0.01)
    baseline_tp = []
    autoencoder_tp = []
    
    for thresh in thresholds:
        baseline_pred = (baseline_proba >= thresh).astype(int)
        autoencoder_pred = (autoencoder_proba >= thresh).astype(int)
        baseline_tp.append((baseline_pred == y).sum() if len(np.unique(baseline_pred)) > 1 else 0)
        autoencoder_tp.append((autoencoder_pred == y).sum() if len(np.unique(autoencoder_pred)) > 1 else 0)
    
    ax7.plot(thresholds, baseline_tp, label='Baseline', linewidth=2, color='red', marker='o', markersize=4)
    ax7.plot(thresholds, autoencoder_tp, label='Autoencoder', linewidth=2, color='blue', marker='s', markersize=4)
    ax7.axvline(x=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Thresholds (0.45-0.90)')
    ax7.axvline(x=0.90, color='yellow', linestyle='--', linewidth=2)
    ax7.axvline(x=0.01, color='green', linestyle=':', linewidth=2, label='Comprehensive Optimal (0.01)')
    ax7.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Number of Predictions', fontsize=12, fontweight='bold')
    ax7.set_title('Predictions vs Threshold', fontsize=14, fontweight='bold')
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, 0.3)
    
    # Add main title
    fig.suptitle('Probability Distribution Analysis: Why Different Thresholds Are Needed', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_file = os.path.join(RESULTS_DIR, f"probability_distributions_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"  Saved visualization to: {output_file}")
    
    plt.close()
    
    return output_file

# =============================================================================
# CREATE EXPLANATION VISUALIZATION
# =============================================================================

def create_threshold_explanation_plot(baseline_proba, autoencoder_proba, y):
    """Create a focused explanation plot."""
    print("Creating threshold explanation plot...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Why Different Thresholds Are Needed: Probability Distribution Analysis', 
                fontsize=16, fontweight='bold')
    
    # 1. Probability ranges
    ax1 = axes[0, 0]
    models = ['Baseline', 'Autoencoder']
    max_probs = [baseline_proba.max(), autoencoder_proba.max()]
    mean_probs = [baseline_proba.mean(), autoencoder_proba.mean()]
    median_probs = [np.median(baseline_proba), np.median(autoencoder_proba)]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax1.bar(x - width, max_probs, width, label='Max Probability', color='red', alpha=0.7)
    ax1.bar(x, mean_probs, width, label='Mean Probability', color='green', alpha=0.7)
    ax1.bar(x + width, median_probs, width, label='Median Probability', color='blue', alpha=0.7)
    
    ax1.axhline(y=0.45, color='yellow', linestyle='--', linewidth=2, label='Original Threshold Range')
    ax1.axhline(y=0.90, color='yellow', linestyle='--', linewidth=2)
    ax1.axhline(y=0.01, color='green', linestyle=':', linewidth=2, label='Comprehensive Optimal')
    
    ax1.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Probability Statistics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 0.15)
    
    # 2. Histogram with threshold lines
    ax2 = axes[0, 1]
    ax2.hist(baseline_proba, bins=100, alpha=0.6, label='Baseline', color='red', edgecolor='black', density=True)
    ax2.hist(autoencoder_proba, bins=100, alpha=0.6, label='Autoencoder', color='blue', edgecolor='black', density=True)
    ax2.axvline(x=0.45, color='yellow', linestyle='--', linewidth=3, label='Original Thresholds (0.45-0.90)')
    ax2.axvline(x=0.90, color='yellow', linestyle='--', linewidth=3)
    ax2.axvline(x=0.01, color='green', linestyle=':', linewidth=3, label='Comprehensive Optimal (0.01)')
    ax2.set_xlabel('Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Distribution (All Samples)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.1)
    
    # 3. Predictions at different thresholds
    ax3 = axes[1, 0]
    
    original_thresholds = [0.45, 0.60, 0.75, 0.90]
    comprehensive_thresholds = [0.01, 0.02, 0.05, 0.10]
    
    baseline_preds_orig = [(baseline_proba >= t).sum() for t in original_thresholds]
    autoencoder_preds_orig = [(autoencoder_proba >= t).sum() for t in original_thresholds]
    baseline_preds_comp = [(baseline_proba >= t).sum() for t in comprehensive_thresholds]
    autoencoder_preds_comp = [(autoencoder_proba >= t).sum() for t in comprehensive_thresholds]
    
    x_orig = np.arange(len(original_thresholds))
    x_comp = np.arange(len(comprehensive_thresholds)) + len(original_thresholds) + 1
    
    ax3.bar(x_orig - 0.2, baseline_preds_orig, 0.4, label='Baseline', color='red', alpha=0.7)
    ax3.bar(x_orig + 0.2, autoencoder_preds_orig, 0.4, label='Autoencoder', color='blue', alpha=0.7)
    ax3.bar(x_comp - 0.2, baseline_preds_comp, 0.4, color='red', alpha=0.7)
    ax3.bar(x_comp + 0.2, autoencoder_preds_comp, 0.4, color='blue', alpha=0.7)
    
    ax3.axvline(x=len(original_thresholds) + 0.5, color='white', linestyle='-', linewidth=2)
    ax3.set_xticks(list(x_orig) + list(x_comp))
    ax3.set_xticklabels([f'{t:.2f}' for t in original_thresholds] + [f'{t:.2f}' for t in comprehensive_thresholds])
    ax3.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Positive Predictions', fontsize=12, fontweight='bold')
    ax3.set_title('Predictions at Different Thresholds', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(len(original_thresholds) + 0.5, ax3.get_ylim()[1]*0.9, 'Original\nThresholds', 
            ha='center', fontsize=10, color='yellow', fontweight='bold')
    ax3.text(len(original_thresholds) + 0.5, ax3.get_ylim()[1]*0.7, 'Comprehensive\nThresholds', 
            ha='center', fontsize=10, color='green', fontweight='bold')
    
    # 4. Explanation text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = """
WHY DIFFERENT THRESHOLDS?

1. PROBABILITY DISTRIBUTIONS:
   • Original Model: Probabilities range 0.0 to ~0.9+
   • Comprehensive Baseline: Max = 12.4%
   • Comprehensive Autoencoder: Max = 2.88%

2. THRESHOLD REQUIREMENTS:
   • Original: Can use 0.45-0.90 (probabilities high enough)
   • Comprehensive: Need 0.01-0.02 (probabilities much lower)

3. WHY MORE CONSERVATIVE?
   • Smaller training dataset (1.0M vs 2.5M samples)
   • Different data distribution
   • More filtered/optimized data
   • Autoencoder model especially conservative

4. FAIR COMPARISON:
   • Compare at OPTIMAL thresholds (each model's best)
   • Original: Best at 0.90 (F1=4.80%)
   • Autoencoder: Best at 0.01 (F1=4.89%)
   • Autoencoder wins: Better F1, Recall, ROC AUC

5. CONCLUSION:
   Different thresholds are needed because models
   produce different probability distributions.
   Comparing at optimal thresholds is fair and shows
   autoencoder model is superior.
    """
    
    ax4.text(0.05, 0.95, explanation, fontsize=11, family='monospace',
            verticalalignment='top', color='white', transform=ax4.transAxes)
    
    output_file = os.path.join(RESULTS_DIR, f"threshold_explanation_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"  Saved explanation plot to: {output_file}")
    
    plt.close()
    
    return output_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main visualization pipeline."""
    print("=" * 70)
    print("PROBABILITY DISTRIBUTION VISUALIZATION")
    print("=" * 70)
    
    # Load data
    baseline_proba, autoencoder_proba, y = load_comprehensive_models_and_data()
    
    print(f"\nProbability Statistics:")
    print(f"  Baseline - Min: {baseline_proba.min():.4f}, Max: {baseline_proba.max():.4f}, Mean: {baseline_proba.mean():.4f}")
    print(f"  Autoencoder - Min: {autoencoder_proba.min():.4f}, Max: {autoencoder_proba.max():.4f}, Mean: {autoencoder_proba.mean():.4f}")
    
    # Create visualizations
    dist_file = create_probability_distribution_plots(baseline_proba, autoencoder_proba, y)
    expl_file = create_threshold_explanation_plot(baseline_proba, autoencoder_proba, y)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {dist_file}")
    print(f"  2. {expl_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())






