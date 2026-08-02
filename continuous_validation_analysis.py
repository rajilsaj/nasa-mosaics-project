#!/usr/bin/env python3
"""
Continuous Validation Analysis for Vortex Detection
==================================================

This script creates continuous twinx plots showing:
- Pressure data (left y-axis, black line)
- Model confidence for precursor prediction (right y-axis, red line)
- All ground truth regions (4xFWHM, detection, FWHM)
- Model predictions and threshold

Optimized with:
- Pre-computed features (no retraining)
- Efficient data loading
- Fast visualization generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
VALIDATION_ML_FILE = "datasets/temporal_splits/ml_val.csv"
VALIDATION_SLIDING_FEATURES_FILE = "datasets/val_sliding_features.csv"
TRAIN_FEATURES_FILE = "datasets/train_features.csv"
OUTPUT_DIR = "continuous_validation_analysis"

# =============================================================================
# Helper Functions
# =============================================================================

def load_data():
    """Load validation data and train model efficiently."""
    print("Loading data...")
    
    # Load validation ML data
    val_ml = pd.read_csv(VALIDATION_ML_FILE)
    print(f"  Loaded {len(val_ml):,} validation ML samples")
    
    # Load pre-computed sliding features
    val_features = pd.read_csv(VALIDATION_SLIDING_FEATURES_FILE)
    print(f"  Loaded {len(val_features):,} pre-computed sliding window features")
    
    # Train model quickly on training features
    train_features = pd.read_csv(TRAIN_FEATURES_FILE)
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    # Fast model training
    rf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced for speed
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training model (fast version)...")
    rf_model.fit(X_train, y_train)
    print("  Model ready!")
    
    return val_ml, val_features, rf_model, feature_cols

def get_model_predictions(model, features_df, feature_cols):
    """Get model predictions and probabilities from pre-computed features."""
    print("Getting model predictions...")
    
    # Filter valid predictions (exclude 'Omit' labels)
    valid_features = features_df[features_df['label'] != 'Omit'].copy()
    valid_features['label'] = valid_features['label'].map({'True': 1, 'False': 0})
    
    X_features = valid_features[feature_cols].values
    
    # Get probabilities
    probabilities = model.predict_proba(X_features)[:, 1]
    
    # Use 0.5 threshold for predictions
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)
    
    print(f"  Generated {len(probabilities):,} predictions")
    print(f"  Positive predictions: {predictions.sum():,}")
    
    return probabilities, predictions, valid_features

def align_predictions_with_time_series(val_ml, val_features, probabilities, predictions):
    """Align sliding window predictions with continuous time series."""
    print("Aligning predictions with time series...")
    
    # Create continuous arrays for the full time series
    continuous_probs = np.zeros(len(val_ml))
    continuous_preds = np.zeros(len(val_ml))
    
    # Map sliding window predictions to time series
    valid_idx = 0  # Index for valid predictions array
    for idx, row in val_features.iterrows():
        if row['label'] != 'Omit':  # Only process valid predictions
            start_idx = int(row['start_idx'])
            end_idx = int(row['end_idx'])
            
            # Ensure indices are within bounds
            if 0 <= end_idx < len(continuous_probs) and valid_idx < len(probabilities):
                continuous_probs[end_idx] = probabilities[valid_idx]
                continuous_preds[end_idx] = predictions[valid_idx]
                valid_idx += 1
    
    # Forward fill to create continuous signal
    # This creates a step-like function where predictions persist until updated
    continuous_probs = pd.Series(continuous_probs).fillna(method='ffill').fillna(0).values
    continuous_preds = pd.Series(continuous_preds).fillna(method='ffill').fillna(0).values
    
    print(f"  Created continuous signals: {len(continuous_probs):,} time points")
    print(f"  Processed {valid_idx:,} valid predictions")
    
    return continuous_probs, continuous_preds

def find_ground_truth_regions(df):
    """Find all ground truth regions in the data."""
    print("Finding ground truth regions...")
    
    regions = {}
    
    # Find 4xFWHM regions
    regions['fourxfwhm'] = find_contiguous_regions(df, 'gt_4xfwhm')
    
    # Find detection regions
    regions['detection'] = find_contiguous_regions(df, 'gt_detection_win')
    
    # Find FWHM regions
    regions['fwhm'] = find_contiguous_regions(df, 'gt_fwhm')
    
    print(f"  Found {len(regions['fourxfwhm'])} 4xFWHM regions")
    print(f"  Found {len(regions['detection'])} detection regions")
    print(f"  Found {len(regions['fwhm'])} FWHM regions")
    
    return regions

def find_contiguous_regions(df, column_name):
    """Find contiguous regions where column is True."""
    if column_name not in df.columns:
        return []
    
    regions = []
    in_region = False
    start_idx = None
    
    for idx, row in df.iterrows():
        if row[column_name] == True:
            if not in_region:
                start_idx = idx
                in_region = True
        else:
            if in_region:
                # End of region
                if start_idx is not None:
                    regions.append((start_idx, idx - 1))
                in_region = False
                start_idx = None
    
    # Handle case where region extends to end
    if in_region and start_idx is not None:
        regions.append((start_idx, len(df) - 1))
    
    return regions

def create_continuous_validation_plot(val_ml, continuous_probs, continuous_preds, gt_regions, 
                                   sclk_start=None, sclk_end=None, output_path=None):
    """
    Create a twinx continuous validation plot showing:
    - SCLK on x-axis
    - Pressure on left y-axis (black line)
    - Model confidence on right y-axis (red line)
    - All ground truth regions
    - Model predictions overlaid
    """
    
    # Filter by SCLK range if specified
    if sclk_start is not None:
        mask1 = val_ml['SCLK'] >= sclk_start
    else:
        mask1 = np.ones(len(val_ml), dtype=bool)
    
    if sclk_end is not None:
        mask2 = val_ml['SCLK'] <= sclk_end
    else:
        mask2 = np.ones(len(val_ml), dtype=bool)
    
    mask = mask1 & mask2
    plot_df = val_ml[mask].copy().reset_index(drop=True)
    plot_probs = continuous_probs[mask]
    plot_preds = continuous_preds[mask]
    
    if len(plot_df) == 0:
        print("No data in the specified SCLK range")
        return
    
    # Create the twinx plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax2 = ax.twinx()
    
    # LEFT Y-AXIS: Pressure (black line)
    ax.plot(plot_df['SCLK'], plot_df['PRESSURE'], 'k-', 
            alpha=0.7, linewidth=1, label='Pressure')
    ax.set_xlabel('SCLK', fontsize=12)
    ax.set_ylabel('Pressure (Pa)', color='black', fontsize=12)
    ax.tick_params(axis='y', labelcolor='black')
    
    # RIGHT Y-AXIS: Model Confidence (red line)
    ax2.plot(plot_df['SCLK'], plot_probs, 'r-', 
             alpha=0.8, linewidth=2, label='Model Confidence')
    ax2.set_ylabel('Model Confidence (0-1)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    # Confidence threshold line
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (0.5)')
    
    # Ground truth regions
    # 4xFWHM regions (Gray background)
    for i, (start, end) in enumerate(gt_regions['fourxfwhm']):
        if start < len(plot_df) and end < len(plot_df):
            start_sclk = plot_df.iloc[start]['SCLK']
            end_sclk = plot_df.iloc[end]['SCLK']
            label = 'GT 4xFWHM Window' if i == 0 else ''
            ax.axvspan(start_sclk, end_sclk, alpha=0.15, color='gray', zorder=1, label=label)
    
    # Detection regions (Red background)
    for i, (start, end) in enumerate(gt_regions['detection']):
        if start < len(plot_df) and end < len(plot_df):
            start_sclk = plot_df.iloc[start]['SCLK']
            end_sclk = plot_df.iloc[end]['SCLK']
            label = 'GT Detection Window' if i == 0 else ''
            ax.axvspan(start_sclk, end_sclk, alpha=0.25, color='red', zorder=1, label=label)
    
    # FWHM regions (Green background)
    for i, (start, end) in enumerate(gt_regions['fwhm']):
        if start < len(plot_df) and end < len(plot_df):
            start_sclk = plot_df.iloc[start]['SCLK']
            end_sclk = plot_df.iloc[end]['SCLK']
            label = 'GT FWHM Window' if i == 0 else ''
            ax.axvspan(start_sclk, end_sclk, alpha=0.25, color='green', zorder=1, label=label)
    
    # Model predictions as orange scatter points
    pred_mask = plot_preds == 1
    if pred_mask.any():
        ax2.scatter(plot_df.loc[pred_mask, 'SCLK'], 
                   plot_probs[pred_mask], 
                   color='orange', s=30, alpha=0.9, zorder=3, 
                   label='Model Predictions', edgecolors='white', linewidth=0.5)
    
    # Set title and grid
    title = 'Continuous Validation Analysis: Pressure vs Model Confidence vs Ground Truth'
    if sclk_start is not None and sclk_end is not None:
        title += f'\nSCLK Range: {sclk_start:.1f} - {sclk_end:.1f}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Remove duplicate labels
    all_lines = []
    all_labels = []
    seen_labels = set()
    
    for line, label in zip(lines1 + lines2, labels1 + labels2):
        if label not in seen_labels:
            all_lines.append(line)
            all_labels.append(label)
            seen_labels.add(label)
    
    ax.legend(all_lines, all_labels, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Auto-fit y-axis for pressure
    pressure_mean = plot_df['PRESSURE'].mean()
    pressure_std = plot_df['PRESSURE'].std()
    ax.set_ylim(pressure_mean - 3 * pressure_std, pressure_mean + 3 * pressure_std)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Continuous validation plot saved to: {output_path}")
    
    plt.show()
    return fig

def analyze_validation_performance(val_ml, continuous_probs, continuous_preds):
    """Analyze the validation performance statistics."""
    
    print("\n" + "="*60)
    print("CONTINUOUS VALIDATION ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"Total time points: {len(val_ml):,}")
    print(f"SCLK range: {val_ml['SCLK'].min():.1f} to {val_ml['SCLK'].max():.1f}")
    print(f"Pressure range: {val_ml['PRESSURE'].min():.3f} to {val_ml['PRESSURE'].max():.3f} Pa")
    
    # Model confidence statistics
    print(f"\nModel Confidence Statistics:")
    print(f"  Mean: {continuous_probs.mean():.4f}")
    print(f"  Std: {continuous_probs.std():.4f}")
    print(f"  Min: {continuous_probs.min():.4f}")
    print(f"  Max: {continuous_probs.max():.4f}")
    
    # Prediction statistics
    positive_predictions = continuous_preds.sum()
    print(f"\nPrediction Statistics:")
    print(f"  Positive predictions: {positive_predictions:,} ({positive_predictions/len(continuous_preds)*100:.2f}%)")
    print(f"  Negative predictions: {len(continuous_preds) - positive_predictions:,}")
    
    # Ground truth statistics
    if 'gt_detection_win' in val_ml.columns:
        detection_count = val_ml['gt_detection_win'].sum()
        print(f"\nGround Truth Statistics:")
        print(f"  GT Detection Windows: {detection_count:,} samples")
    
    if 'gt_fwhm' in val_ml.columns:
        fwhm_count = val_ml['gt_fwhm'].sum()
        print(f"  GT FWHM Events: {fwhm_count:,} samples")
    
    if 'gt_4xfwhm' in val_ml.columns:
        fourxfwhm_count = val_ml['gt_4xfwhm'].sum()
        print(f"  GT 4xFWHM Windows: {fourxfwhm_count:,} samples")

# =============================================================================
# Main Execution
# =============================================================================

def run_continuous_validation_analysis(sclk_start=None, sclk_end=None):
    """
    Run the complete continuous validation analysis.
    
    Args:
        sclk_start: Start SCLK for focused analysis (optional)
        sclk_end: End SCLK for focused analysis (optional)
    """
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("CONTINUOUS VALIDATION ANALYSIS")
    print("="*60)
    
    # 1. Load data and train model
    val_ml, val_features, model, feature_cols = load_data()
    
    # 2. Get model predictions
    probabilities, predictions, valid_features = get_model_predictions(model, val_features, feature_cols)
    
    # 3. Align predictions with time series
    continuous_probs, continuous_preds = align_predictions_with_time_series(val_ml, val_features, probabilities, predictions)
    
    # 4. Find ground truth regions
    gt_regions = find_ground_truth_regions(val_ml)
    
    # 5. Analyze performance
    analyze_validation_performance(val_ml, continuous_probs, continuous_preds)
    
    # 6. Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full validation plot
    full_plot_path = os.path.join(OUTPUT_DIR, f"continuous_validation_full_{timestamp}.png")
    create_continuous_validation_plot(
        val_ml, continuous_probs, continuous_preds, gt_regions,
        sclk_start, sclk_end, full_plot_path
    )
    
    # If SCLK range specified, also create a zoomed plot
    if sclk_start is not None and sclk_end is not None:
        zoom_plot_path = os.path.join(OUTPUT_DIR, f"continuous_validation_zoom_{timestamp}.png")
        create_continuous_validation_plot(
            val_ml, continuous_probs, continuous_preds, gt_regions,
            sclk_start, sclk_end, zoom_plot_path
        )
        print(f"\nZoomed plot ({sclk_start:.1f} - {sclk_end:.1f}) saved to: {zoom_plot_path}")
    
    print(f"\nAnalysis complete!")
    print(f"Plots saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for SCLK range (optional)
    sclk_start = None
    sclk_end = None
    
    if len(sys.argv) > 1:
        try:
            if len(sys.argv) >= 3:
                sclk_start = float(sys.argv[1])
                sclk_end = float(sys.argv[2])
                print(f"Analyzing SCLK range: {sclk_start} to {sclk_end}")
            else:
                sclk_start = float(sys.argv[1])
                print(f"Analyzing from SCLK {sclk_start} to end of data")
                sclk_end = None
        except ValueError:
            print("Usage: python continuous_validation_analysis.py [sclk_start] [sclk_end]")
            print("SCLK values should be floating point numbers")
            sys.exit(1)
    
    run_continuous_validation_analysis(sclk_start, sclk_end)
