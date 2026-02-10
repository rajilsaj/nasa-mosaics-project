#!/usr/bin/env python3
"""
Feature Importance and Confusion Matrix Visualization
=======================================================

Creates:
1. Feature importance plots for baseline and autoencoder models
2. Confusion matrices on test set at optimal thresholds
"""

import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Optimal thresholds
BASELINE_THRESHOLD = 0.02
AUTOENCODER_THRESHOLD = 0.01

# Set style
plt.style.use('default')
sns.set_palette("husl")

# =============================================================================
# LOAD MODELS AND DATA
# =============================================================================

def load_models_and_data():
    """Load models and test sliding window features."""
    print("=" * 70)
    print("LOADING MODELS AND TEST DATA")
    print("=" * 70)
    
    # Load models
    baseline_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_model_*.pkl"))
    autoencoder_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_*.pkl"))
    
    if not baseline_files or not autoencoder_files:
        print("[ERROR] Models not found!")
        return None, None, None, None, None, None, None
    
    baseline_model = joblib.load(max(baseline_files, key=os.path.getctime))
    autoencoder_model = joblib.load(max(autoencoder_files, key=os.path.getctime))
    
    print(f"Loaded baseline model: {os.path.basename(max(baseline_files, key=os.path.getctime))}")
    print(f"Loaded autoencoder model: {os.path.basename(max(autoencoder_files, key=os.path.getctime))}")
    
    # Load test sliding window features
    features_file = os.path.join(FEATURES_DIR, "test_sliding_features_step10.csv")
    if not os.path.exists(features_file):
        print(f"[ERROR] Test features not found: {features_file}")
        return None, None, None, None, None, None, None
    
    features_df = pd.read_csv(features_file)
    print(f"Loaded {len(features_df):,} test feature vectors")
    
    # Load model metadata to get correct feature columns
    baseline_metadata_files = glob.glob(os.path.join(MODELS_DIR, "baseline_rf_metadata_*.json"))
    autoencoder_metadata_files = glob.glob(os.path.join(MODELS_DIR, "rf_with_autoencoder_metadata_*.json"))
    
    if baseline_metadata_files:
        with open(max(baseline_metadata_files, key=os.path.getctime), 'r') as f:
            baseline_metadata = json.load(f)
            baseline_features = baseline_metadata.get('features', [])
    else:
        baseline_features = []
    
    if autoencoder_metadata_files:
        with open(max(autoencoder_metadata_files, key=os.path.getctime), 'r') as f:
            autoencoder_metadata = json.load(f)
            autoencoder_features = autoencoder_metadata.get('features', [])
    else:
        autoencoder_features = []
    
    # Separate features and labels
    label_cols = ['label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx', 
                  'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in features_df.columns if col not in label_cols]
    
    # Select features for each model
    baseline_feature_cols = [f for f in baseline_features if f in feature_cols]
    autoencoder_feature_cols = [f for f in autoencoder_features if f in feature_cols]
    
    baseline_X = features_df[baseline_feature_cols].values
    autoencoder_X = features_df[autoencoder_feature_cols].values
    y = features_df['label'].values
    
    print(f"\nBaseline features: {len(baseline_feature_cols)}")
    print(f"Autoencoder features: {len(autoencoder_feature_cols)}")
    
    return (baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, 
            baseline_feature_cols, autoencoder_feature_cols)

# =============================================================================
# FEATURE IMPORTANCE PLOTS
# =============================================================================

def create_feature_importance_plots(baseline_model, autoencoder_model, 
                                    baseline_features, autoencoder_features):
    """Create feature importance plots for both models."""
    print("\n" + "=" * 70)
    print("CREATING FEATURE IMPORTANCE PLOTS")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get feature importances
    baseline_importance = baseline_model.feature_importances_
    autoencoder_importance = autoencoder_model.feature_importances_
    
    # Create DataFrames
    baseline_df = pd.DataFrame({
        'feature': baseline_features,
        'importance': baseline_importance
    }).sort_values('importance', ascending=False)
    
    autoencoder_df = pd.DataFrame({
        'feature': autoencoder_features,
        'importance': autoencoder_importance
    }).sort_values('importance', ascending=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Feature Importance Comparison: Baseline vs Autoencoder Models', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Baseline model plot
    ax1 = axes[0]
    top_baseline = baseline_df.head(15)
    colors = ['red' if 'autoencoder' not in feat.lower() else 'blue' 
              for feat in top_baseline['feature']]
    ax1.barh(range(len(top_baseline)), top_baseline['importance'], color=colors)
    ax1.set_yticks(range(len(top_baseline)))
    ax1.set_yticklabels(top_baseline['feature'], fontsize=10)
    ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline Model (15 features)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_baseline.iterrows()):
        ax1.text(row['importance'] + 0.001, i, f"{row['importance']:.4f}", 
                va='center', fontsize=9)
    
    # Autoencoder model plot
    ax2 = axes[1]
    top_autoencoder = autoencoder_df.head(19)
    colors = ['red' if 'autoencoder' not in feat.lower() else 'blue' 
              for feat in top_autoencoder['feature']]
    ax2.barh(range(len(top_autoencoder)), top_autoencoder['importance'], color=colors)
    ax2.set_yticks(range(len(top_autoencoder)))
    ax2.set_yticklabels(top_autoencoder['feature'], fontsize=10)
    ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Autoencoder Model (19 features)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_autoencoder.iterrows()):
        ax2.text(row['importance'] + 0.001, i, f"{row['importance']:.4f}", 
                va='center', fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Original Features'),
        Patch(facecolor='blue', label='Autoencoder Features')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(RESULTS_DIR, f"feature_importance_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved feature importance plot to: {output_file}")
    
    plt.close()
    
    # Create combined top features comparison
    fig2, ax = plt.subplots(figsize=(14, 10))
    
    # Get top 10 from each
    top_baseline_10 = baseline_df.head(10)
    top_autoencoder_10 = autoencoder_df.head(10)
    
    # Create comparison
    x = np.arange(len(top_autoencoder_10))
    width = 0.35
    
    ax.barh(x - width/2, top_baseline_10['importance'].values, width, 
           label='Baseline', color='red', alpha=0.7)
    ax.barh(x + width/2, top_autoencoder_10['importance'].values, width, 
           label='Autoencoder', color='blue', alpha=0.7)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_autoencoder_10['feature'], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Feature Importance Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    output_file2 = os.path.join(RESULTS_DIR, f"feature_importance_comparison_{timestamp}.png")
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"  Saved feature importance comparison to: {output_file2}")
    
    plt.close()
    
    return baseline_df, autoencoder_df

# =============================================================================
# CONFUSION MATRIX PLOTS
# =============================================================================

def create_confusion_matrices(baseline_model, autoencoder_model, 
                              baseline_X, autoencoder_X, y):
    """Create confusion matrices for both models on test set."""
    print("\n" + "=" * 70)
    print("CREATING CONFUSION MATRICES")
    print("=" * 70)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get predictions at optimal thresholds
    baseline_proba = baseline_model.predict_proba(baseline_X)[:, 1]
    autoencoder_proba = autoencoder_model.predict_proba(autoencoder_X)[:, 1]
    
    baseline_pred = (baseline_proba >= BASELINE_THRESHOLD).astype(int)
    autoencoder_pred = (autoencoder_proba >= AUTOENCODER_THRESHOLD).astype(int)
    
    # Calculate confusion matrices
    baseline_cm = confusion_matrix(y, baseline_pred)
    autoencoder_cm = confusion_matrix(y, autoencoder_pred)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Confusion Matrices on Test Set (Sliding Windows)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Baseline confusion matrix
    ax1 = axes[0]
    sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title(f'Baseline Model (Threshold: {BASELINE_THRESHOLD})', 
                 fontsize=14, fontweight='bold')
    
    # Add metrics
    tn, fp, fn, tp = baseline_cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"Precision: {precision:.2%}\nRecall: {recall:.2%}\nF1-Score: {f1:.2%}"
    ax1.text(0.5, -0.15, metrics_text, transform=ax1.transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Autoencoder confusion matrix
    ax2 = axes[1]
    sns.heatmap(autoencoder_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title(f'Autoencoder Model (Threshold: {AUTOENCODER_THRESHOLD})', 
                 fontsize=14, fontweight='bold')
    
    # Add metrics
    tn, fp, fn, tp = autoencoder_cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"Precision: {precision:.2%}\nRecall: {recall:.2%}\nF1-Score: {f1:.2%}"
    ax2.text(0.5, -0.15, metrics_text, transform=ax2.transAxes,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(RESULTS_DIR, f"confusion_matrices_{timestamp}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved confusion matrices to: {output_file}")
    
    plt.close()
    
    # Print detailed metrics
    tn, fp, fn, tp = baseline_cm.ravel()
    baseline_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    baseline_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0
    
    print("\nBaseline Model Confusion Matrix:")
    print(f"  TN: {baseline_cm[0,0]}, FP: {baseline_cm[0,1]}")
    print(f"  FN: {baseline_cm[1,0]}, TP: {baseline_cm[1,1]}")
    print(f"  Precision: {baseline_precision:.2%}, Recall: {baseline_recall:.2%}, F1: {baseline_f1:.2%}")
    
    tn, fp, fn, tp = autoencoder_cm.ravel()
    autoencoder_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    autoencoder_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    autoencoder_f1 = 2 * (autoencoder_precision * autoencoder_recall) / (autoencoder_precision + autoencoder_recall) if (autoencoder_precision + autoencoder_recall) > 0 else 0
    
    print("\nAutoencoder Model Confusion Matrix:")
    print(f"  TN: {autoencoder_cm[0,0]}, FP: {autoencoder_cm[0,1]}")
    print(f"  FN: {autoencoder_cm[1,0]}, TP: {autoencoder_cm[1,1]}")
    print(f"  Precision: {autoencoder_precision:.2%}, Recall: {autoencoder_recall:.2%}, F1: {autoencoder_f1:.2%}")
    
    return baseline_cm, autoencoder_cm

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main visualization pipeline."""
    print("=" * 70)
    print("FEATURE IMPORTANCE AND CONFUSION MATRIX VISUALIZATION")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models and data
    result = load_models_and_data()
    if result[0] is None:
        return 1
    
    (baseline_model, autoencoder_model, baseline_X, autoencoder_X, y, 
     baseline_features, autoencoder_features) = result
    
    # Create feature importance plots
    baseline_df, autoencoder_df = create_feature_importance_plots(
        baseline_model, autoencoder_model, baseline_features, autoencoder_features
    )
    
    # Create confusion matrices
    baseline_cm, autoencoder_cm = create_confusion_matrices(
        baseline_model, autoencoder_model, baseline_X, autoencoder_X, y
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETED")
    print("=" * 70)
    print("\nFiles created in results/ directory:")
    print("  - feature_importance_*.png")
    print("  - feature_importance_comparison_*.png")
    print("  - confusion_matrices_*.png")
    
    return 0

if __name__ == "__main__":
    exit(main())

