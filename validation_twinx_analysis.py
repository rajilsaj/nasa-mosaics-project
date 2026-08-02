#!/usr/bin/env python3
"""
Validation Set Twinx Analysis
Visualizes raw pressure data vs Random Forest probability predictions on validation set
Similar to training_twinx_analysis.py but for validation sliding windows
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def load_validation_data_and_model():
    """Load validation ML data, sliding features, and trained model."""
    print("=" * 70)
    print("VALIDATION SET TWINX ANALYSIS")
    print("=" * 70)
    
    # Load validation ML data (continuous time series)
    val_ml_df = pd.read_csv("datasets/temporal_splits/ml_val.csv")
    print(f"Loaded {len(val_ml_df):,} validation ML samples")
    print(f"SCLK range: {val_ml_df['SCLK'].min():.2f} - {val_ml_df['SCLK'].max():.2f}")
    
    # Load validation sliding features
    val_features_df = pd.read_csv("datasets/val_sliding_features.csv")
    print(f"Loaded {len(val_features_df):,} validation sliding window feature vectors")
    
    # Find latest model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) 
                   if f.startswith("improved_rf_vortex_detector_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No improved model found. Run improved_train_rf_model.py first.")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    print(f"Loading model: {latest_model}")
    
    # Load model
    model = joblib.load(model_path)
    
    return val_ml_df, val_features_df, model

def get_model_predictions(val_features_df, model):
    """Get model probability predictions for validation sliding windows."""
    print("\nGetting model predictions for validation windows...")
    
    # Prepare features (match training feature order)
    train_df = pd.read_csv("datasets/train_features.csv")
    train_feature_cols = [col for col in train_df.columns 
                          if col not in ['window_id', 'label', 'event_sclk']]
    
    # Filter valid predictions (exclude 'Omit' labels)
    valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
    
    feature_cols = [col for col in train_feature_cols 
                   if col in valid_df.columns]
    
    X_val = valid_df[feature_cols].values
    
    # Get model probabilities
    rf_prob = model.predict_proba(X_val)[:, 1]
    
    print(f"Generated {len(rf_prob):,} predictions")
    print(f"Probability range: {rf_prob.min():.4f} - {rf_prob.max():.4f}")
    
    # Add probabilities to valid_df
    valid_df = valid_df.copy()
    valid_df['rf_prob'] = rf_prob
    
    return valid_df

def align_predictions_with_timeseries(val_ml_df, valid_df):
    """Align sliding window predictions with continuous time series."""
    print("\nAligning predictions with continuous time series...")
    
    # Create arrays for continuous time series
    continuous_probs = np.full(len(val_ml_df), np.nan)
    window_ids = np.full(len(val_ml_df), -1, dtype=int)
    
    # Map sliding window predictions to time series using end_idx
    for idx, row in valid_df.iterrows():
        end_idx = int(row['end_idx'])
        if 0 <= end_idx < len(val_ml_df):
            continuous_probs[end_idx] = row['rf_prob']
            # Store window_id for reference
            if 'window_id' in row:
                window_ids[end_idx] = row['window_id']
    
    # Forward fill probabilities to create continuous signal
    # This creates a step-like function where predictions persist until updated
    continuous_probs_series = pd.Series(continuous_probs)
    continuous_probs_filled = continuous_probs_series.ffill().fillna(0).values
    
    print(f"Aligned {np.sum(~np.isnan(continuous_probs)):,} window predictions")
    print(f"Created continuous probability signal for {len(val_ml_df):,} time points")
    
    return continuous_probs_filled, continuous_probs  # Return both filled and original

def plot_validation_twinx(val_ml_df, continuous_probs, output_dir="results", save_plot=True):
    """Create twinx plot showing pressure vs RF probability for validation set."""
    
    fig, ax1 = plt.subplots(figsize=(15, 8))

    # --- Pressure (left y-axis)
    ax1.plot(val_ml_df['SCLK'], val_ml_df['PRESSURE'], 'k.', markersize=1, alpha=0.6, label='Pressure')
    ax1.set_xlabel("SCLK (Time)", fontsize=12)
    ax1.set_ylabel("Pressure (Pa)", color='k', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='k')

    # --- Shaded ground-truth regions (temporal sequence: precursor → vortex)
    # Plot 4xFWHM first (background - extended region)
    if 'gt_4xfwhm' in val_ml_df.columns and val_ml_df['gt_4xfwhm'].any():
        mask = val_ml_df['gt_4xfwhm'].astype(bool)
        print(f"Plotting 4xFWHM Region: {mask.sum()} samples")
        ax1.fill_between(val_ml_df['SCLK'],
                         val_ml_df['PRESSURE'].min(),
                         val_ml_df['PRESSURE'].max(),
                         where=mask, color='lightgray', alpha=0.15, label='4xFWHM (Extended Region)')
    
    # Plot detection window (PRECURSOR - comes BEFORE vortex)
    if 'gt_detection_win' in val_ml_df.columns and val_ml_df['gt_detection_win'].any():
        mask = val_ml_df['gt_detection_win'].astype(bool)
        print(f"Plotting Detection Window (Precursor): {mask.sum()} samples")
        ax1.fill_between(val_ml_df['SCLK'],
                         val_ml_df['PRESSURE'].min(),
                         val_ml_df['PRESSURE'].max(),
                         where=mask, color='red', alpha=0.4, label='Precursor Region (Detection Window)')
    
    # Plot FWHM last (ACTUAL VORTEX - comes AFTER precursor)
    if 'gt_fwhm' in val_ml_df.columns and val_ml_df['gt_fwhm'].any():
        mask = val_ml_df['gt_fwhm'].astype(bool)
        print(f"Plotting FWHM Region (Actual Vortex): {mask.sum()} samples")
        ax1.fill_between(val_ml_df['SCLK'],
                         val_ml_df['PRESSURE'].min(),
                         val_ml_df['PRESSURE'].max(),
                         where=mask, color='green', alpha=0.7, label='Actual Vortex (FWHM)')

    # --- RF Probability (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(val_ml_df['SCLK'], continuous_probs, color='tab:blue', lw=1.5, alpha=0.8, label='RF Probability')
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

    plt.title("Validation Set: Pressure vs RF Probability (Precursor → Vortex Sequence)\n" + 
              f"Red = Precursor Regions, Green = Actual Vortex Events\n" +
              f"({len(val_ml_df):,} samples, {np.sum(~np.isnan(continuous_probs)):,} window predictions)", 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(output_dir, f"validation_twinx_analysis_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nValidation twinx plot saved to: {plot_filename}")
    
    plt.show()
    
    return fig

def analyze_validation_predictions(val_ml_df, continuous_probs, valid_df):
    """Analyze model predictions on validation data."""
    print("\n" + "=" * 70)
    print("VALIDATION SET PREDICTION ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    valid_predictions = valid_df['rf_prob']
    print(f"Valid predictions: {len(valid_predictions):,} windows")
    print(f"Probability range: {valid_predictions.min():.4f} - {valid_predictions.max():.4f}")
    print(f"Mean probability: {valid_predictions.mean():.4f}")
    print(f"Std probability: {valid_predictions.std():.4f}")
    
    # Analysis by ground truth regions
    for col, label in zip(['gt_4xfwhm', 'gt_detection_win', 'gt_fwhm'], 
                         ['4xFWHM (Extended)', 'Precursor Region', 'Actual Vortex']):
        if col in val_ml_df.columns:
            region_mask = val_ml_df[col].astype(bool)
            region_probs = continuous_probs[region_mask]
            region_probs = region_probs[~np.isnan(region_probs)]
            
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
        print(f"  Threshold {threshold}: {above_threshold:,} windows ({percentage:.1f}%)")
    
    # Label distribution
    if 'label' in valid_df.columns:
        print(f"\nLabel Distribution:")
        label_counts = valid_df['label'].value_counts()
        for label, count in label_counts.items():
            print(f"  {label}: {count:,} windows")

def main():
    """Main execution function."""
    print("Starting validation set twinx analysis...")
    
    # Step 1: Load validation data and model
    val_ml_df, val_features_df, model = load_validation_data_and_model()
    
    # Step 2: Get model predictions
    valid_df = get_model_predictions(val_features_df, model)
    
    # Step 3: Align predictions with continuous time series
    continuous_probs_filled, continuous_probs_original = align_predictions_with_timeseries(val_ml_df, valid_df)
    
    # Step 4: Create twinx plot
    plot_validation_twinx(val_ml_df, continuous_probs_filled)
    
    # Step 5: Analyze predictions
    analyze_validation_predictions(val_ml_df, continuous_probs_original, valid_df)
    
    print("\n" + "=" * 70)
    print("VALIDATION SET TWINX ANALYSIS COMPLETED")
    print("=" * 70)
    print("Key Insights:")
    print("- Visual correlation between pressure patterns and model confidence")
    print("- Ground truth regions provide context for model behavior")
    print("- Multiple threshold lines show different operating points")
    print("- Validation performance on continuous monitoring scenario")
    print("=" * 70)

if __name__ == "__main__":
    main()

