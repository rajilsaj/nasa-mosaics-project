#!/usr/bin/env python3
"""
Sliding Window Probability Visualization
=======================================

This script creates a twinx plot showing:
- Left Y-axis: Raw pressure data (black dots)
- Right Y-axis: RF model's precursor probability (red curve)
- Background: Ground truth regions (gray/red/green)
- X-axis: Time progression

The RF model is applied to overlapping sliding windows (step=20) to create
continuous-looking probability curves that validate the model's understanding
of temporal vortex physics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Feature engineering functions (copied to avoid import issues)
def compute_trend_features(pressure_values):
    """Compute trend features for pressure decrease detection."""
    features = {}
    
    # Overall slope (primary signal)
    x = np.arange(len(pressure_values))
    slope, _ = np.polyfit(x, pressure_values, 1)
    features['overall_slope'] = slope
    
    # First half slope
    h = len(pressure_values) // 2
    x_first = np.arange(h)
    slope_first, _ = np.polyfit(x_first, pressure_values[:h], 1)
    features['first_half_slope'] = slope_first
    
    # Second half slope
    x_second = np.arange(h)
    slope_second, _ = np.polyfit(x_second, pressure_values[h:], 1)
    features['second_half_slope'] = slope_second
    
    # Trend consistency (difference between halves)
    features['trend_consistency'] = abs(slope_first - slope_second)
    
    return features

def compute_pressure_drop_features(pressure_values):
    """Compute pressure drop magnitude and characteristics."""
    features = {}
    
    # Pressure drop (start to minimum)
    pressure_drop = pressure_values[0] - np.min(pressure_values)
    features['pressure_drop'] = pressure_drop
    
    # Drop rate (pressure drop per sample)
    min_idx = np.argmin(pressure_values)
    if min_idx > 0:
        features['drop_rate'] = pressure_drop / min_idx
    else:
        features['drop_rate'] = 0.0
    
    # Position of minimum (normalized)
    features['min_position'] = min_idx / len(pressure_values)
    
    return features

def compute_core_statistics(pressure_values):
    """Compute basic statistical features."""
    features = {}
    
    features['mean'] = np.mean(pressure_values)
    features['std'] = np.std(pressure_values)
    features['range'] = np.max(pressure_values) - np.min(pressure_values)
    
    return features

def compute_temporal_evolution_features(pressure_values):
    """Compute temporal evolution features."""
    features = {}
    
    h = len(pressure_values) // 2
    
    # First and second half means
    features['first_half_mean'] = np.mean(pressure_values[:h])
    features['second_half_mean'] = np.mean(pressure_values[h:])
    
    # Mean ratio (second/first)
    if features['first_half_mean'] != 0:
        features['mean_ratio'] = features['second_half_mean'] / features['first_half_mean']
    else:
        features['mean_ratio'] = 1.0
    
    return features

def compute_anomaly_features(pressure_values, global_mean, global_std):
    """Compute anomaly detection features."""
    features = {}
    
    # Minimum z-score (how unusual is the lowest pressure?)
    min_pressure = np.min(pressure_values)
    features['min_zscore'] = (min_pressure - global_mean) / global_std
    
    # Anomaly strength (how much does the minimum deviate?)
    features['anomaly_strength'] = abs(features['min_zscore'])
    
    return features

def load_model_and_data():
    """Load the trained RF model and validation data."""
    print("Loading model and data...")
    
    # Train model directly (avoid pickle issues)
    from sklearn.ensemble import RandomForestClassifier
    
    # Load training features
    train_features = pd.read_csv('datasets/train_features.csv')
    print(f"  Loaded training features: {len(train_features):,} samples")
    
    # Prepare training data
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    print(f"  Training features: {len(feature_cols)}")
    print(f"  Training samples: {len(X_train)}")
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("  Model training completed!")
    
    # Load validation ML data
    val_ml = pd.read_csv('datasets/temporal_splits/ml_val.csv')
    print(f"  Loaded validation data: {len(val_ml):,} samples")
    
    # Get global statistics
    global_mean = train_features['mean'].mean()
    global_std = train_features['std'].mean()
    
    print(f"  Global mean: {global_mean:.4f}")
    print(f"  Global std: {global_std:.4f}")
    
    return rf_model, val_ml, global_mean, global_std, feature_cols

def engineer_features_for_window(window_data, global_mean, global_std):
    """Engineer features for a single window (same as training)."""
    if len(window_data) < 60:
        return None
    
    pressure = window_data['PRESSURE'].values
    
    features = {}
    
    # Trend features
    trend_feats = compute_trend_features(pressure)
    features.update(trend_feats)
    
    # Pressure drop features
    drop_feats = compute_pressure_drop_features(pressure)
    features.update(drop_feats)
    
    # Core statistics
    stats_feats = compute_core_statistics(pressure)
    features.update(stats_feats)
    
    # Temporal evolution
    temporal_feats = compute_temporal_evolution_features(pressure)
    features.update(temporal_feats)
    
    # Anomaly detection
    anomaly_feats = compute_anomaly_features(pressure, global_mean, global_std)
    features.update(anomaly_feats)
    
    return features

def create_sliding_windows(ml_data, rf_model, feature_cols, global_mean, global_std, window_size=60, step_size=20):
    """Create sliding windows and compute probabilities."""
    print(f"\nCreating sliding windows...")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    
    total_windows = (len(ml_data) - window_size) // step_size + 1
    print(f"  Total windows: {total_windows:,}")
    
    probabilities = []
    window_positions = []
    
    for i in tqdm(range(0, len(ml_data) - window_size + 1, step_size), 
                  desc="Processing windows"):
        # Extract window
        window_data = ml_data.iloc[i:i + window_size].copy()
        
        # Engineer features
        features = engineer_features_for_window(window_data, global_mean, global_std)
        
        if features is not None:
            # Prepare feature vector (same order as training)
            feature_vector = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            
            # Get probability
            prob = rf_model.predict_proba(feature_vector)[0, 1]  # Probability of positive class
            
            probabilities.append(prob)
            window_positions.append(i + window_size - 1)  # Right edge of window
    
    return np.array(probabilities), np.array(window_positions)

def create_ground_truth_regions(ml_data):
    """Create ground truth region boundaries."""
    print("\nCreating ground truth regions...")
    
    # Find regions where gt_detection_win = True
    detection_regions = []
    detection_start = None
    
    for i, row in ml_data.iterrows():
        if row['gt_detection_win'] == True:
            if detection_start is None:
                detection_start = i
        else:
            if detection_start is not None:
                detection_regions.append((detection_start, i-1))
                detection_start = None
    
    # Handle case where region extends to end
    if detection_start is not None:
        detection_regions.append((detection_start, len(ml_data)-1))
    
    # Find regions where gt_fwhm = True
    fwhm_regions = []
    fwhm_start = None
    
    for i, row in ml_data.iterrows():
        if row['gt_fwhm'] == True:
            if fwhm_start is None:
                fwhm_start = i
        else:
            if fwhm_start is not None:
                fwhm_regions.append((fwhm_start, i-1))
                fwhm_start = None
    
    if fwhm_start is not None:
        fwhm_regions.append((fwhm_start, len(ml_data)-1))
    
    print(f"  Detection regions: {len(detection_regions)}")
    print(f"  FWHM regions: {len(fwhm_regions)}")
    
    return detection_regions, fwhm_regions

def create_twinx_plot(ml_data, probabilities, window_positions, detection_regions, fwhm_regions, output_file):
    """Create the twinx visualization plot."""
    print(f"\nCreating twinx plot...")
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Left Y-axis: Pressure data
    sample_indices = np.arange(len(ml_data))
    pressure_values = ml_data['PRESSURE'].values
    
    # Plot pressure data (every 10th point for performance)
    step = max(1, len(ml_data) // 10000)  # Limit to ~10k points
    ax1.scatter(sample_indices[::step], pressure_values[::step], 
               c='black', s=0.5, alpha=0.6, label='Pressure')
    
    ax1.set_xlabel('Sample Index', fontsize=12)
    ax1.set_ylabel('Pressure (Pa)', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Right Y-axis: Model probabilities
    ax2 = ax1.twinx()
    ax2.plot(window_positions, probabilities, 'r-', linewidth=2, 
             label='RF Precursor Probability', alpha=0.8)
    ax2.set_ylabel('Precursor Probability', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    # Add probability threshold line
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold (0.5)')
    
    # Add ground truth regions
    y_min, y_max = ax1.get_ylim()
    
    # Detection regions (red background)
    for start, end in detection_regions:
        ax1.axvspan(start, end, alpha=0.2, color='red', label='gt_detection_win' if start == detection_regions[0][0] else "")
    
    # FWHM regions (green background)
    for start, end in fwhm_regions:
        ax1.axvspan(start, end, alpha=0.2, color='green', label='gt_fwhm' if start == fwhm_regions[0][0] else "")
    
    # Set title and legend
    plt.title('Random Forest Vortex Detection: Continuous Probability Analysis\n' +
              f'Sliding Windows (size=60, step=20) - {len(probabilities):,} windows', 
              fontsize=14, pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved plot to: {output_file}")
    
    return fig

def analyze_model_behavior(ml_data, probabilities, window_positions, detection_regions, fwhm_regions):
    """Analyze the model's behavior across different regions."""
    print(f"\nAnalyzing model behavior...")
    
    # Convert to arrays for easier analysis
    probs = np.array(probabilities)
    positions = np.array(window_positions)
    
    # Analyze detection regions
    detection_probs = []
    for start, end in detection_regions:
        # Find probabilities within this region
        mask = (positions >= start) & (positions <= end)
        if np.any(mask):
            region_probs = probs[mask]
            detection_probs.extend(region_probs)
            print(f"  Detection region {start}-{end}: {len(region_probs)} windows, "
                  f"mean prob: {np.mean(region_probs):.3f}")
    
    # Analyze FWHM regions
    fwhm_probs = []
    for start, end in fwhm_regions:
        mask = (positions >= start) & (positions <= end)
        if np.any(mask):
            region_probs = probs[mask]
            fwhm_probs.extend(region_probs)
            print(f"  FWHM region {start}-{end}: {len(region_probs)} windows, "
                  f"mean prob: {np.mean(region_probs):.3f}")
    
    # Overall statistics
    print(f"\nOverall statistics:")
    print(f"  Total windows: {len(probabilities):,}")
    print(f"  Mean probability: {np.mean(probabilities):.3f}")
    print(f"  Std probability: {np.std(probabilities):.3f}")
    print(f"  Min probability: {np.min(probabilities):.3f}")
    print(f"  Max probability: {np.max(probabilities):.3f}")
    
    if detection_probs:
        print(f"  Detection regions - Mean: {np.mean(detection_probs):.3f}, "
              f"Std: {np.std(detection_probs):.3f}")
    
    if fwhm_probs:
        print(f"  FWHM regions - Mean: {np.mean(fwhm_probs):.3f}, "
              f"Std: {np.std(fwhm_probs):.3f}")

def main():
    """Main execution function."""
    print("="*70)
    print("SLIDING WINDOW PROBABILITY VISUALIZATION")
    print("="*70)
    
    try:
        # Load model and data
        rf_model, val_ml, global_mean, global_std, feature_cols = load_model_and_data()
        
        # Create sliding windows and compute probabilities
        probabilities, window_positions = create_sliding_windows(
            val_ml, rf_model, feature_cols, global_mean, global_std, window_size=60, step_size=20)
        
        # Create ground truth regions
        detection_regions, fwhm_regions = create_ground_truth_regions(val_ml)
        
        # Create twinx plot
        output_file = "sliding_window_probability_analysis.png"
        fig = create_twinx_plot(val_ml, probabilities, window_positions, 
                               detection_regions, fwhm_regions, output_file)
        
        # Analyze model behavior
        analyze_model_behavior(val_ml, probabilities, window_positions, 
                              detection_regions, fwhm_regions)
        
        print(f"\n{'='*70}")
        print("SLIDING WINDOW ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Visualization saved to: {output_file}")
        print(f"Total sliding windows processed: {len(probabilities):,}")
        print(f"Step size: 20 samples")
        print(f"Window size: 60 samples")
        
        plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
