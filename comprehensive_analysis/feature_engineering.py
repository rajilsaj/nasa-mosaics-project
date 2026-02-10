#!/usr/bin/env python3
"""
Feature Engineering for Comprehensive Dataset with Autoencoder Features
=======================================================================

As a seasoned RF scientist, this script engineers:
1. 15 proven features from pressure data (baseline)
2. 3-5 autoencoder-derived features (incremental addition)
3. Total: 18-20 optimized features for Random Forest

Key Principles:
- Start with proven features (don't break what works)
- Add autoencoder incrementally (measure contribution)
- Handle edge cases robustly
- Maintain temporal causality
- Optimize for on-board inference
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_SIZE = 60
INPUT_DIR = "data/windows"
OUTPUT_DIR = "data/features"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Engineer features from windows for comprehensive dataset',
        epilog="""
Examples:
  python feature_engineering.py --split train
  python feature_engineering.py --split val --include_autoencoder
  python feature_engineering.py --split test --include_autoencoder
        """
    )
    
    parser.add_argument('--split', 
                       choices=['train', 'val', 'test'],
                       required=True,
                       help='Which temporal split to process')
    
    parser.add_argument('--include_autoencoder',
                       action='store_true',
                       help='Include autoencoder features (default: False for baseline)')
    
    parser.add_argument('--window_size',
                       type=int,
                       default=60,
                       help='Window size in samples (default: 60)')
    
    return parser.parse_args()

# =============================================================================
# FEATURE ENGINEERING FUNCTIONS (15 PROVEN FEATURES)
# =============================================================================

def compute_trend_features(pressure_values, window_size=60):
    """
    Compute trend features for pressure decrease detection.
    
    These capture the temporal evolution of pressure, which is critical
    for detecting vortex precursor patterns.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        window_size: Size of window (default: 60)
        
    Returns:
        Dictionary with 4 trend features
    """
    features = {}
    
    if len(pressure_values) < 2:
        return {
            'overall_slope': 0.0,
            'first_half_slope': 0.0,
            'second_half_slope': 0.0,
            'trend_consistency': 0.0
        }
    
    # Overall slope (primary signal - linear trend across entire window)
    x = np.arange(len(pressure_values))
    try:
        slope, intercept = np.polyfit(x, pressure_values, 1)
        features['overall_slope'] = float(slope)
    except:
        features['overall_slope'] = 0.0
    
    # First half slope (trend in first 30 samples)
    first_half = pressure_values[:window_size//2]
    if len(first_half) >= 2:
        x_first = np.arange(len(first_half))
        try:
            slope_first, _ = np.polyfit(x_first, first_half, 1)
            features['first_half_slope'] = float(slope_first)
        except:
            features['first_half_slope'] = 0.0
    else:
        features['first_half_slope'] = 0.0
    
    # Second half slope (trend in last 30 samples)
    second_half = pressure_values[window_size//2:]
    if len(second_half) >= 2:
        x_second = np.arange(len(second_half))
        try:
            slope_second, _ = np.polyfit(x_second, second_half, 1)
            features['second_half_slope'] = float(slope_second)
        except:
            features['second_half_slope'] = 0.0
    else:
        features['second_half_slope'] = 0.0
    
    # Trend consistency (how similar are the slopes?)
    # Lower value = more consistent trend (better for vortex detection)
    features['trend_consistency'] = float(abs(features['first_half_slope'] - 
                                             features['second_half_slope']))
    
    return features

def compute_pressure_drop_features(pressure_values, window_size=60):
    """
    Compute pressure drop magnitude and timing features.
    
    Vortex events cause characteristic pressure drops. These features
    quantify the magnitude and temporal position of the drop.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        window_size: Size of window (default: 60)
        
    Returns:
        Dictionary with 3 pressure drop features
    """
    features = {}
    
    if len(pressure_values) == 0:
        return {
            'pressure_drop': 0.0,
            'drop_rate': 0.0,
            'min_position': 0.5
        }
    
    # Total pressure drop (max - min)
    # Vortex events show significant pressure drops
    pressure_drop = float(pressure_values.max() - pressure_values.min())
    features['pressure_drop'] = pressure_drop
    
    # Drop rate (pressure drop per sample to minimum)
    # Faster drops are more characteristic of vortices
    min_idx = int(np.argmin(pressure_values))
    if min_idx > 0:
        # Rate of drop from start to minimum
        drop_rate = pressure_drop / min_idx
    else:
        drop_rate = 0.0
    features['drop_rate'] = float(drop_rate)
    
    # Minimum position (where does the drop occur?)
    # Normalized to [0, 1] - 0 = start, 1 = end
    # Vortex drops typically occur in second half
    min_position = float(min_idx) / (window_size - 1) if window_size > 1 else 0.5
    features['min_position'] = min_position
    
    return features

def compute_core_statistics(pressure_values):
    """
    Compute core statistical features.
    
    Basic statistical properties that capture window characteristics.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        
    Returns:
        Dictionary with 2 statistical features (range removed - duplicate of pressure_drop)
    """
    features = {}
    
    if len(pressure_values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0
        }
    
    features['mean'] = float(np.mean(pressure_values))
    features['std'] = float(np.std(pressure_values)) if len(pressure_values) > 1 else 0.0
    # NOTE: 'range' feature removed - it's identical to 'pressure_drop' (Phase 2 fix)
    
    return features

def compute_temporal_evolution(pressure_values, window_size=60):
    """
    Compute temporal evolution features (first vs second half comparison).
    
    Captures how pressure changes between window halves, which is
    characteristic of vortex precursor patterns.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        window_size: Size of window (default: 60)
        
    Returns:
        Dictionary with 3 temporal evolution features
    """
    features = {}
    
    if len(pressure_values) < window_size//2:
        return {
            'first_half_mean': 0.0,
            'second_half_mean': 0.0,
            'mean_ratio': 1.0
        }
    
    # Split into halves
    first_half = pressure_values[:window_size//2]
    second_half = pressure_values[window_size//2:]
    
    # Half means
    first_half_mean = float(np.mean(first_half))
    second_half_mean = float(np.mean(second_half))
    
    features['first_half_mean'] = first_half_mean
    features['second_half_mean'] = second_half_mean
    
    # Mean ratio (how much does pressure change between halves?)
    # Ratio < 1.0 indicates pressure drop (vortex characteristic)
    if first_half_mean != 0:
        mean_ratio = second_half_mean / first_half_mean
    else:
        mean_ratio = 1.0
    features['mean_ratio'] = float(mean_ratio)
    
    return features

def compute_anomaly_features(pressure_values, global_mean=None, global_std=None):
    """
    Compute anomaly detection features for rare event identification.
    
    Vortex events are rare anomalies. These features quantify how
    extreme the pressure values are relative to normal conditions.
    
    Args:
        pressure_values: Array of pressure values (60 samples)
        global_mean: Global pressure mean (for z-score normalization)
        global_std: Global pressure std (for z-score normalization)
        
    Returns:
        Dictionary with 2 anomaly features
    """
    features = {}
    
    if len(pressure_values) == 0:
        return {
            'min_zscore': 0.0,
            'anomaly_strength': 0.0
        }
    
    # Min z-score (how extreme is the minimum pressure?)
    # Negative z-score = below normal (vortex characteristic)
    if global_mean is not None and global_std is not None and global_std > 0:
        min_zscore = (np.min(pressure_values) - global_mean) / global_std
    else:
        # Use local statistics if global not available
        local_mean = np.mean(pressure_values)
        local_std = np.std(pressure_values)
        if local_std > 0:
            min_zscore = (np.min(pressure_values) - local_mean) / local_std
        else:
            min_zscore = 0.0
    features['min_zscore'] = float(min_zscore)
    
    # Anomaly strength (magnitude of deviation from mean)
    # Higher values = more anomalous (potential vortex)
    mean_pressure = np.mean(pressure_values)
    std_pressure = np.std(pressure_values)
    if std_pressure > 0:
        anomaly_strength = abs(np.min(pressure_values) - mean_pressure) / std_pressure
    else:
        anomaly_strength = 0.0
    features['anomaly_strength'] = float(anomaly_strength)
    
    return features

# =============================================================================
# AUTOENCODER FEATURE ENGINEERING
# =============================================================================

def compute_autoencoder_features(window_data):
    """
    Compute autoencoder-derived features.
    
    The autoencoder provides complementary signal to pressure-based features.
    These features capture autoencoder patterns.
    
    PHASE 2 FIX: 
    - Removed ae_gt_agreement (uses ground truth - data leakage)
    - Added proper NaN handling to prevent perfect separation
    - Ensures identical processing for positive and negative windows
    
    Args:
        window_data: DataFrame with window data (includes autoencoder columns)
        
    Returns:
        Dictionary with 3 autoencoder features (ae_gt_agreement removed)
    """
    features = {}
    
    # Check if autoencoder columns exist
    if 'autoencoder_window_hits' not in window_data.columns:
        # No autoencoder features available - return NaN (not 0.0) to avoid perfect separation
        return {
            'autoencoder_window_hits_mean': np.nan,
            'autoencoder_positive_hit_binary': np.nan,
            'autoencoder_hit_ratio': np.nan
        }
    
    # Extract autoencoder data
    ae_hits = window_data['autoencoder_window_hits'].values
    
    # PHASE 2 FIX: Check for NaN values in the data itself
    # If all values are NaN, return NaN (don't create perfect separation)
    if np.all(np.isnan(ae_hits)):
        return {
            'autoencoder_window_hits_mean': np.nan,
            'autoencoder_positive_hit_binary': np.nan,
            'autoencoder_hit_ratio': np.nan
        }
    
    # Mean autoencoder hits in window (handle NaN values)
    # Higher values = more autoencoder confidence
    ae_hits_clean = ae_hits[~np.isnan(ae_hits)]
    if len(ae_hits_clean) > 0:
        features['autoencoder_window_hits_mean'] = float(np.mean(ae_hits_clean))
    else:
        features['autoencoder_window_hits_mean'] = np.nan
    
    # Binary: Does autoencoder detect any positive hit in window?
    # 1 = autoencoder detected vortex, 0 = no detection
    if 'autoencoder_positive_hit' in window_data.columns:
        ae_positive = window_data['autoencoder_positive_hit'].values
        # Handle NaN values
        ae_positive_clean = ae_positive[~np.isnan(ae_positive)]
        if len(ae_positive_clean) > 0:
            features['autoencoder_positive_hit_binary'] = float(1.0 if np.any(ae_positive_clean > 0) else 0.0)
        else:
            features['autoencoder_positive_hit_binary'] = np.nan
    else:
        features['autoencoder_positive_hit_binary'] = np.nan
    
    # Hit ratio: Total hits / window size
    # Normalized confidence measure [0, 1]
    if len(ae_hits_clean) > 0:
        total_hits = np.sum(ae_hits_clean)
        features['autoencoder_hit_ratio'] = float(total_hits / WINDOW_SIZE)
    else:
        features['autoencoder_hit_ratio'] = np.nan
    
    # PHASE 2 FIX: Removed ae_gt_agreement - it uses ground truth (data leakage)
    # This feature compared autoencoder output with gt_detection_win, which is the label!
    
    return features

# =============================================================================
# MAIN FEATURE ENGINEERING FUNCTION
# =============================================================================

def engineer_features_for_window(window_data, global_mean=None, global_std=None, 
                                 include_autoencoder=False, window_size=60):
    """
    Engineer all features for a single window.
    
    This is the main feature engineering function that combines:
    - 15 proven pressure-based features
    - 3-5 autoencoder features (if available)
    
    Args:
        window_data: DataFrame with pressure data for one window (60 rows)
        global_mean: Global pressure mean (for z-score normalization)
        global_std: Global pressure std (for z-score normalization)
        include_autoencoder: Whether to include autoencoder features
        window_size: Size of window (default: 60)
        
    Returns:
        Dictionary with all engineered features
    """
    # Validate window
    if window_data is None or len(window_data) == 0:
        raise ValueError("Window data is empty")
    
    if len(window_data) < window_size:
        raise ValueError(f"Window too small: {len(window_data)} < {window_size}")
    
    # Extract pressure values
    if 'PRESSURE' not in window_data.columns:
        raise ValueError("PRESSURE column not found in window data")
    
    pressure_values = window_data['PRESSURE'].values
    
    # Combine all feature groups
    features = {}
    
    # 14 Proven Features (Baseline) - range removed (duplicate of pressure_drop)
    features.update(compute_trend_features(pressure_values, window_size))
    features.update(compute_pressure_drop_features(pressure_values, window_size))
    features.update(compute_core_statistics(pressure_values))  # Now returns 2 features (mean, std)
    features.update(compute_temporal_evolution(pressure_values, window_size))
    features.update(compute_anomaly_features(pressure_values, global_mean, global_std))
    
    # Autoencoder Features (Incremental Addition)
    if include_autoencoder:
        ae_features = compute_autoencoder_features(window_data)
        features.update(ae_features)
    
    # Add metadata (preserve for traceability)
    if 'window_id' in window_data.columns:
        features['window_id'] = int(window_data['window_id'].iloc[0])
    if 'event_sclk' in window_data.columns:
        features['event_sclk'] = float(window_data['event_sclk'].iloc[0])
    if 'label' in window_data.columns:
        # Convert boolean to int if needed
        label_val = window_data['label'].iloc[0]
        if isinstance(label_val, bool):
            features['label'] = int(label_val)
        else:
            features['label'] = int(label_val)
    
    return features

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def calculate_global_statistics(split_name):
    """
    Calculate global pressure statistics from the full split (not just windows).
    
    These are used for z-score normalization in anomaly features.
    
    Args:
        split_name: 'train', 'val', or 'test'
        
    Returns:
        Tuple of (global_mean, global_std)
    """
    split_file = os.path.join("data/splits", f"ml_{split_name}.csv")
    
    if not os.path.exists(split_file):
        print(f"  [WARNING] Split file not found: {split_file}")
        print(f"  [INFO] Using local statistics for anomaly features")
        return None, None
    
    try:
        split_df = pd.read_csv(split_file)
        global_mean = float(split_df['PRESSURE'].mean())
        global_std = float(split_df['PRESSURE'].std())
        print(f"  Global statistics: mean={global_mean:.2f} Pa, std={global_std:.2f} Pa")
        return global_mean, global_std
    except Exception as e:
        print(f"  [WARNING] Could not calculate global statistics: {e}")
        return None, None

def main():
    """Main feature engineering pipeline."""
    args = parse_arguments()
    
    print("=" * 70)
    print(f"FEATURE ENGINEERING - {args.split.upper()} SPLIT")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Split: {args.split}")
    print(f"  Include autoencoder: {args.include_autoencoder}")
    print(f"  Window size: {args.window_size}")
    
    # Input/Output files
    input_file = os.path.join(INPUT_DIR, f"{args.split}_windows.csv")
    output_file = os.path.join(OUTPUT_DIR, f"{args.split}_features.csv")
    
    # Check input file exists
    if not os.path.exists(input_file):
        print(f"\n[ERROR] Input file not found: {input_file}")
        return 1
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load windows
    print(f"\nLoading windows from: {input_file}")
    windows_df = pd.read_csv(input_file)
    print(f"  Loaded {len(windows_df):,} rows")
    
    # Count unique windows
    if 'window_id' in windows_df.columns:
        num_windows = windows_df['window_id'].nunique()
        print(f"  Unique windows: {num_windows}")
    else:
        print(f"  [WARNING] No window_id column found")
        num_windows = len(windows_df) // args.window_size
    
    # Calculate global statistics (for z-score normalization)
    print(f"\nCalculating global statistics...")
    global_mean, global_std = calculate_global_statistics(args.split)
    
    # Check for autoencoder features
    has_autoencoder = ('autoencoder_window_hits' in windows_df.columns and 
                      'autoencoder_positive_hit' in windows_df.columns)
    
    if args.include_autoencoder and not has_autoencoder:
        print(f"  [WARNING] Autoencoder features requested but not found in windows")
        print(f"  [INFO] Proceeding with baseline features only")
        args.include_autoencoder = False
    
    if has_autoencoder and not args.include_autoencoder:
        print(f"  [INFO] Autoencoder features available but not included (baseline mode)")
    
    # Engineer features for each window
    print(f"\nEngineering features...")
    print(f"  Baseline features: 14 (range removed - duplicate of pressure_drop)")
    if args.include_autoencoder:
        print(f"  Autoencoder features: 3 (ae_gt_agreement removed - data leakage)")
        print(f"  Total features: 17")
    else:
        print(f"  Total features: 14")
    
    all_features = []
    failed_windows = 0
    
    # Group by window_id and process each window
    if 'window_id' in windows_df.columns:
        window_groups = windows_df.groupby('window_id')
    else:
        # Fallback: assume windows are sequential
        print(f"  [WARNING] No window_id found, assuming sequential windows")
        window_groups = []
        for i in range(0, len(windows_df), args.window_size):
            window_data = windows_df.iloc[i:i+args.window_size]
            window_groups.append((i // args.window_size, window_data))
    
    for window_id, window_data in tqdm(window_groups, desc="Processing windows"):
        try:
            features = engineer_features_for_window(
                window_data,
                global_mean=global_mean,
                global_std=global_std,
                include_autoencoder=args.include_autoencoder,
                window_size=args.window_size
            )
            all_features.append(features)
        except Exception as e:
            print(f"  [ERROR] Failed to engineer features for window {window_id}: {e}")
            failed_windows += 1
            continue
    
    if len(all_features) == 0:
        print(f"\n[ERROR] No features were successfully engineered!")
        return 1
    
    # Convert to DataFrame
    features_df = pd.DataFrame(all_features)
    
    # Feature count validation
    baseline_feature_count = 14  # Updated: range removed
    expected_features = baseline_feature_count
    if args.include_autoencoder:
        expected_features += 3  # 3 autoencoder features (ae_gt_agreement removed)
    
    actual_features = len([col for col in features_df.columns 
                          if col not in ['window_id', 'event_sclk', 'label']])
    
    print(f"\nFeature engineering results:")
    print(f"  Successfully processed: {len(all_features)} windows")
    print(f"  Failed: {failed_windows} windows")
    print(f"  Features created: {actual_features}")
    print(f"  Expected: {expected_features}")
    
    if actual_features < expected_features:
        print(f"  [WARNING] Fewer features than expected!")
    
    # Save features
    print(f"\nSaving features to: {output_file}")
    features_df.to_csv(output_file, index=False)
    print(f"  [SUCCESS] Saved {len(features_df):,} rows to {output_file}")
    
    # Feature summary
    print(f"\nFeature summary:")
    feature_cols = [col for col in features_df.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Features: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
    
    if 'label' in features_df.columns:
        label_counts = features_df['label'].value_counts()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  Label {label}: {count} ({count/len(features_df)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())

