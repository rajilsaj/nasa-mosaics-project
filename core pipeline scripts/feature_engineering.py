"""
Feature Engineering for Mars Vortex Detection
==============================================

This script engineers 15 optimized features for Random Forest vortex detection:
1. Trend Features (4) - Primary signal detection
2. Pressure Drop Features (3) - Magnitude quantification  
3. Core Statistics (3) - Baseline properties
4. Temporal Evolution (3) - Pattern recognition
5. Anomaly Detection (2) - Rare event identification

Features are designed for on-board inference on Qualcomm Snapdragon processors.
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LinearRegression

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments for split processing."""
    parser = argparse.ArgumentParser(
        description='Engineer features from balanced windows',
        epilog="""
Examples:
  python feature_engineering.py --split train
  python feature_engineering.py --split val
  python feature_engineering.py --split test
        """
    )
    
    parser.add_argument('--split', 
                       choices=['train', 'val', 'test'],
                       required=True,
                       help='Which temporal split to process')
    
    parser.add_argument('--window_size',
                       type=int,
                       default=60,
                       help='Window size in samples (default: 60)')
    
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = f"{args.split}_balanced.csv"
OUTPUT_FILE = f"{args.split}_features.csv"
WINDOW_SIZE = args.window_size

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

def compute_trend_features(pressure_values):
    """
    Compute trend features for pressure decrease detection.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        
    Returns:
        Dictionary with trend features
    """
    features = {}
    
    # Overall slope (primary signal)
    x = np.arange(len(pressure_values))
    slope, intercept = np.polyfit(x, pressure_values, 1)
    features['overall_slope'] = float(slope)
    
    # First half slope
    first_half = pressure_values[:WINDOW_SIZE//2]
    x_first = np.arange(len(first_half))
    slope_first, _ = np.polyfit(x_first, first_half, 1)
    features['first_half_slope'] = float(slope_first)
    
    # Second half slope
    second_half = pressure_values[WINDOW_SIZE//2:]
    x_second = np.arange(len(second_half))
    slope_second, _ = np.polyfit(x_second, second_half, 1)
    features['second_half_slope'] = float(slope_second)
    
    # Trend consistency (how similar are the slopes?)
    features['trend_consistency'] = float(abs(slope_first - slope_second))
    
    return features

def compute_pressure_drop_features(pressure_values):
    """
    Compute pressure drop magnitude and timing features.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        
    Returns:
        Dictionary with pressure drop features
    """
    features = {}
    
    # Total pressure drop
    pressure_drop = float(pressure_values[0] - pressure_values.min())
    features['pressure_drop'] = pressure_drop
    
    # Drop rate (pressure drop per sample)
    min_idx = np.argmin(pressure_values)
    if min_idx > 0:
        drop_rate = pressure_drop / min_idx
    else:
        drop_rate = 0.0
    features['drop_rate'] = float(drop_rate)
    
    # Minimum position (where does the drop occur?)
    min_position = float(min_idx) / (WINDOW_SIZE - 1)  # Normalized [0, 1]
    features['min_position'] = min_position
    
    return features

def compute_core_statistics(pressure_values):
    """
    Compute core statistical features.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        
    Returns:
        Dictionary with statistical features
    """
    features = {}
    
    features['mean'] = float(pressure_values.mean())
    features['std'] = float(pressure_values.std())
    features['range'] = float(pressure_values.max() - pressure_values.min())
    
    return features

def compute_temporal_evolution(pressure_values):
    """
    Compute temporal evolution features (first vs second half).
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        
    Returns:
        Dictionary with temporal evolution features
    """
    features = {}
    
    # Split into halves
    first_half = pressure_values[:WINDOW_SIZE//2]
    second_half = pressure_values[WINDOW_SIZE//2:]
    
    # Half means
    first_half_mean = float(first_half.mean())
    second_half_mean = float(second_half.mean())
    
    features['first_half_mean'] = first_half_mean
    features['second_half_mean'] = second_half_mean
    
    # Mean ratio (how much does pressure change between halves?)
    if first_half_mean != 0:
        mean_ratio = second_half_mean / first_half_mean
    else:
        mean_ratio = 1.0
    features['mean_ratio'] = float(mean_ratio)
    
    return features

def compute_anomaly_features(pressure_values, global_mean=None, global_std=None):
    """
    Compute anomaly detection features for rare event identification.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        global_mean: Global pressure mean (for z-score)
        global_std: Global pressure std (for z-score)
        
    Returns:
        Dictionary with anomaly features
    """
    features = {}
    
    # Min z-score (how extreme is the minimum?)
    if global_mean is not None and global_std is not None and global_std > 0:
        min_zscore = (pressure_values.min() - global_mean) / global_std
    else:
        # Use local statistics if global not available
        local_mean = pressure_values.mean()
        local_std = pressure_values.std()
        if local_std > 0:
            min_zscore = (pressure_values.min() - local_mean) / local_std
        else:
            min_zscore = 0.0
    features['min_zscore'] = float(min_zscore)
    
    # Anomaly strength (magnitude of deviation from mean)
    mean_pressure = pressure_values.mean()
    std_pressure = pressure_values.std()
    if std_pressure > 0:
        anomaly_strength = abs(pressure_values.min() - mean_pressure) / std_pressure
    else:
        anomaly_strength = 0.0
    features['anomaly_strength'] = float(anomaly_strength)
    
    return features

def engineer_features_for_window(window_data, global_mean=None, global_std=None):
    """
    Engineer all 15 features for a single window.
    
    Args:
        window_data: DataFrame with pressure data for one window
        global_mean: Global pressure mean (for z-score normalization)
        global_std: Global pressure std (for z-score normalization)
        
    Returns:
        Dictionary with all 15 engineered features
    """
    pressure_values = window_data['PRESSURE'].values
    
    # Combine all feature groups
    features = {}
    features.update(compute_trend_features(pressure_values))
    features.update(compute_pressure_drop_features(pressure_values))
    features.update(compute_core_statistics(pressure_values))
    features.update(compute_temporal_evolution(pressure_values))
    features.update(compute_anomaly_features(pressure_values, global_mean, global_std))
    
    # Add metadata
    features['window_id'] = window_data['window_id'].iloc[0]
    features['event_sclk'] = window_data['event_sclk'].iloc[0]
    
    # Add label (if exists in balanced dataset)
    if 'label' in window_data.columns:
        features['label'] = window_data['label'].iloc[0]
    
    return features

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main feature engineering pipeline."""
    print("="*70)
    print(f"FEATURE ENGINEERING - {args.split.upper()} SPLIT")
    print("="*70)
    
    # Load balanced windows
    print(f"Loading balanced windows from: {INPUT_FILE}")
    windows_df = pd.read_csv(INPUT_FILE)
    print(f"  Loaded {len(windows_df):,} rows")
    print(f"  Unique windows: {windows_df['window_id'].nunique()}")
    
    # Check class distribution
    if 'label' in windows_df.columns:
        class_dist = windows_df.groupby('window_id')['label'].first().value_counts()
        total_windows = windows_df['window_id'].nunique()
        print(f"  Class distribution:")
        print(f"    Positive windows: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total_windows*100:.1f}%)")
        print(f"    Negative windows: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total_windows*100:.1f}%)")
    
    # Compute global statistics for z-score normalization
    # IMPORTANT: Use training split statistics for ALL splits to ensure consistency
    # This prevents distribution shift between train/val/test
    print("\nComputing global statistics for normalization...")
    
    # Try to load training split for consistent statistics
    train_split_file = os.path.join("datasets/temporal_splits", "ml_train.csv")
    if os.path.exists(train_split_file):
        print(f"  Loading training split for consistent global statistics...")
        train_split_df = pd.read_csv(train_split_file)
        global_mean = train_split_df['PRESSURE'].mean()
        global_std = train_split_df['PRESSURE'].std()
        print(f"  Global mean: {global_mean:.4f} Pa (from training split)")
        print(f"  Global std:  {global_std:.4f} Pa (from training split)")
        print(f"  [NOTE] Using training statistics for {args.split} split to ensure consistency")
    else:
        # Fallback: compute from windows (less ideal but works)
        print(f"  [WARNING] Training split not found, using window statistics")
        print(f"  [WARNING] This may cause inconsistency across splits")
        global_mean = windows_df['PRESSURE'].mean()
        global_std = windows_df['PRESSURE'].std()
        print(f"  Global mean: {global_mean:.4f} Pa (from windows)")
        print(f"  Global std:  {global_std:.4f} Pa (from windows)")
    
    # Engineer features for each window
    print(f"\nEngineering features for {windows_df['window_id'].nunique()} windows...")
    
    feature_rows = []
    for window_id, window_data in tqdm(windows_df.groupby('window_id'), 
                                       desc="Feature engineering"):
        features = engineer_features_for_window(window_data, global_mean, global_std)
        feature_rows.append(features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    print(f"\nFeature Engineering Complete!")
    print(f"  Total features: {len(features_df.columns) - 2}")  # Exclude window_id and event_sclk
    print(f"  Feature names: {[col for col in features_df.columns if col not in ['window_id', 'event_sclk']]}")
    
    # Save engineered features
    features_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[SUCCESS] Saved engineered features to: {OUTPUT_FILE}")
    
    # Feature summary
    print(f"\nFeature Summary:")
    print(f"  Trend Features (4): overall_slope, first_half_slope, second_half_slope, trend_consistency")
    print(f"  Pressure Drop (3): pressure_drop, drop_rate, min_position")
    print(f"  Core Statistics (3): mean, std, range")
    print(f"  Temporal Evolution (3): first_half_mean, second_half_mean, mean_ratio")
    print(f"  Anomaly Detection (2): min_zscore, anomaly_strength")
    
    # Quick feature statistics
    print(f"\nFeature Statistics:")
    numeric_cols = [col for col in features_df.columns if col not in ['window_id', 'event_sclk']]
    for col in numeric_cols:
        print(f"  {col:20}: mean={features_df[col].mean():8.4f}, std={features_df[col].std():8.4f}")
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
