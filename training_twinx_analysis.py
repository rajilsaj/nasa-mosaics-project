#!/usr/bin/env python3
"""
Training Set Twinx Analysis
Visualizes raw pressure data vs Random Forest probability predictions
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def load_training_data_and_model():
    """Load training windows data and trained model."""
    print("=" * 70)
    print("TRAINING SET TWINX ANALYSIS")
    print("=" * 70)
    
    # Load raw training windows
    train_df = pd.read_csv("datasets/train_windows.csv")
    print(f"Loaded {len(train_df)} training window samples")
    print(f"Columns: {list(train_df.columns)}")
    
    # Load training features for model predictions
    train_features_df = pd.read_csv("datasets/train_features.csv")
    print(f"Loaded {len(train_features_df)} training feature vectors")
    
    # Find latest model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("improved_rf_vortex_detector_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No improved model found. Run improved_train_rf_model.py first.")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    print(f"Loading model: {latest_model}")
    
    # Load model
    model = joblib.load(model_path)
    
    return train_df, train_features_df, model

def add_model_predictions(train_df, train_features_df, model):
    """Add model probability predictions to training data."""
    print("\nAdding model predictions to training data...")
    
    # Prepare features for prediction (exclude event_sclk to prevent data leakage)
    feature_cols = [col for col in train_features_df.columns if col not in ['window_id', 'label', 'event_sclk']]
    X_train = train_features_df[feature_cols].values
    
    # Get model probabilities
    rf_prob = model.predict_proba(X_train)[:, 1]
    
    # Add probabilities to features dataframe
    train_features_df = train_features_df.copy()
    train_features_df['rf_prob'] = rf_prob
    
    # Create mapping from window_id to probability
    window_prob_map = dict(zip(train_features_df['window_id'], train_features_df['rf_prob']))
    
    # Merge probabilities back to raw training data by window_id
    train_df_with_probs = train_df.copy()
    train_df_with_probs['rf_prob'] = train_df_with_probs['window_id'].map(window_prob_map)
    
    print(f"Added model predictions to {train_df_with_probs['rf_prob'].notna().sum()} samples")
    print(f"Probability range: {train_df_with_probs['rf_prob'].min():.4f} - {train_df_with_probs['rf_prob'].max():.4f}")
    
    return train_df_with_probs

def plot_training_twinx(train_df, time_col='SCLK', pressure_col='PRESSURE', prob_col='rf_prob', 
                       output_dir="results", save_plot=True):
    """Create twinx plot showing pressure vs RF probability."""
    
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # --- Pressure (left y-axis)
    ax1.plot(train_df[time_col], train_df[pressure_col], 'k.', markersize=2, alpha=0.7, label='Pressure')
    ax1.set_xlabel("SCLK (Time)", fontsize=12)
    ax1.set_ylabel("Pressure (Pa)", color='k', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='k')

    # --- Shaded ground-truth regions (temporal sequence: precursor → vortex)
    # Plot 4xFWHM first (background - extended region)
    if 'gt_4xfwhm' in train_df.columns and train_df['gt_4xfwhm'].any():
        mask = train_df['gt_4xfwhm'].astype(bool)
        print(f"Plotting 4xFWHM Region: {mask.sum()} samples")
        ax1.fill_between(train_df[time_col],
                         train_df[pressure_col].min(),
                         train_df[pressure_col].max(),
                         where=mask, color='lightgray', alpha=0.15, label='4xFWHM (Extended Region)')
    
    # Plot detection window (PRECURSOR - comes BEFORE vortex)
    if 'gt_detection_win' in train_df.columns and train_df['gt_detection_win'].any():
        mask = train_df['gt_detection_win'].astype(bool)
        print(f"Plotting Detection Window (Precursor): {mask.sum()} samples")
        ax1.fill_between(train_df[time_col],
                         train_df[pressure_col].min(),
                         train_df[pressure_col].max(),
                         where=mask, color='red', alpha=0.4, label='Precursor Region (Detection Window)')
    
    # Plot FWHM last (ACTUAL VORTEX - comes AFTER precursor)
    if 'gt_fwhm' in train_df.columns and train_df['gt_fwhm'].any():
        mask = train_df['gt_fwhm'].astype(bool)
        print(f"Plotting FWHM Region (Actual Vortex): {mask.sum()} samples")
        ax1.fill_between(train_df[time_col],
                         train_df[pressure_col].min(),
                         train_df[pressure_col].max(),
                         where=mask, color='green', alpha=0.7, label='Actual Vortex (FWHM)')

    # --- RF Probability (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(train_df[time_col], train_df[prob_col], color='tab:blue', lw=1.5, alpha=0.8, label='RF Probability')
    ax2.set_ylabel("Predicted Probability", color='tab:blue', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # Add threshold lines
    ax2.axhline(0.5, color='tab:blue', ls='--', lw=1, alpha=0.6, label='Default Threshold (0.5)')
    ax2.axhline(0.45, color='orange', ls='--', lw=1, alpha=0.6, label='High-Recall Threshold (0.45)')
    ax2.axhline(0.9, color='purple', ls='--', lw=1, alpha=0.6, label='High-Precision Threshold (0.9)')

    # --- Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
               ncol=3, fontsize=10, frameon=True, facecolor='white')

    plt.title("Training Set: Pressure vs RF Probability (Precursor → Vortex Sequence)\n" + 
              f"Red = Precursor Regions, Green = Actual Vortex Events\n" +
              f"({len(train_df):,} samples, {train_df['rf_prob'].notna().sum():,} with predictions)", 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(output_dir, f"training_twinx_analysis_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nTraining twinx plot saved to: {plot_filename}")
    
    plt.show()
    
    return fig

def analyze_training_predictions(train_df):
    """Analyze model predictions on training data."""
    print("\n" + "=" * 70)
    print("TRAINING SET PREDICTION ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    valid_predictions = train_df['rf_prob'].dropna()
    print(f"Valid predictions: {len(valid_predictions):,} / {len(train_df):,} samples")
    print(f"Probability range: {valid_predictions.min():.4f} - {valid_predictions.max():.4f}")
    print(f"Mean probability: {valid_predictions.mean():.4f}")
    print(f"Std probability: {valid_predictions.std():.4f}")
    
    # Analysis by ground truth regions (temporal sequence)
    for col, label in zip(['gt_4xfwhm', 'gt_detection_win', 'gt_fwhm'], 
                         ['4xFWHM (Extended)', 'Precursor Region', 'Actual Vortex']):
        if col in train_df.columns:
            region_mask = train_df[col].astype(bool)
            region_probs = train_df[region_mask]['rf_prob'].dropna()
            
            if len(region_probs) > 0:
                print(f"\n{label}:")
                print(f"  Samples: {len(region_probs):,}")
                print(f"  Mean probability: {region_probs.mean():.4f}")
                print(f"  Max probability: {region_probs.max():.4f}")
                print(f"  Min probability: {region_probs.min():.4f}")
    
    # Threshold analysis
    print(f"\nThreshold Analysis:")
    for threshold in [0.45, 0.5, 0.9]:
        above_threshold = (valid_predictions >= threshold).sum()
        percentage = (above_threshold / len(valid_predictions)) * 100
        print(f"  Threshold {threshold}: {above_threshold:,} samples ({percentage:.1f}%)")

def main():
    """Main execution function."""
    print("Starting training set twinx analysis...")
    
    # Step 1: Load training data and model
    train_df, train_features_df, model = load_training_data_and_model()
    
    # Step 2: Add model predictions
    train_df_with_probs = add_model_predictions(train_df, train_features_df, model)
    
    # Step 3: Create twinx plot
    plot_training_twinx(train_df_with_probs)
    
    # Step 4: Analyze predictions
    analyze_training_predictions(train_df_with_probs)
    
    print("\n" + "=" * 70)
    print("TRAINING SET TWINX ANALYSIS COMPLETED")
    print("=" * 70)
    print("Key Insights:")
    print("- Visual correlation between pressure patterns and model confidence")
    print("- Ground truth regions provide context for model behavior")
    print("- Multiple threshold lines show different operating points")
    print("- Training performance can be visually assessed")
    print("=" * 70)

if __name__ == "__main__":
    main()
