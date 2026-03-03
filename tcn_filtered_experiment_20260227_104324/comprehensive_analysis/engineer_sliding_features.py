#!/usr/bin/env python3
"""
Feature Engineering from Sliding Windows - Comprehensive Dataset
=================================================================

As a seasoned RF scientist, this script:
1. Engineers features from sliding windows (same as fixed windows)
2. Includes autoencoder features if available
3. Processes validation and test sets
4. Saves features for model evaluation

This enables realistic deployment simulation evaluation.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

SLIDING_WINDOWS_DIR = "data/sliding_windows"
FEATURES_DIR = "data/features"
SPLITS_DIR = "data/splits"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Engineer features from sliding windows',
        epilog="""
Examples:
  python engineer_sliding_features.py --split val --step_size 10
  python engineer_sliding_features.py --split test --step_size 10
        """
    )
    
    parser.add_argument('--split', 
                       choices=['val', 'test'],
                       required=True,
                       help='Which temporal split to process')
    
    parser.add_argument('--step_size',
                       type=int,
                       default=10,
                       help='Step size for sliding windows (default: 10)')
    
    parser.add_argument('--window_size',
                       type=int,
                       default=60,
                       help='Window size in samples (default: 60)')
    
    parser.add_argument('--verbose',
                       action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

# =============================================================================
# IMPORT FEATURE ENGINEERING FUNCTIONS
# =============================================================================

# Import from local feature_engineering module
try:
    from feature_engineering import engineer_features_for_window
except ImportError:
    # If import fails, try importing from parent directory
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from feature_engineering import engineer_features_for_window

# =============================================================================
# FEATURE ENGINEERING FROM SLIDING WINDOWS
# =============================================================================

def engineer_features_from_sliding_windows(sliding_windows_file, split_name, 
                                          global_mean, global_std, 
                                          window_size=60, step_size=10,
                                          verbose=False):
    """
    Engineer features from sliding windows.
    
    Args:
        sliding_windows_file: Path to sliding windows CSV file
        split_name: Split name (val or test)
        global_mean: Global pressure mean (from training data)
        global_std: Global pressure std (from training data)
        window_size: Window size in samples
        step_size: Step size for sliding windows
        verbose: Enable verbose output
        
    Returns:
        DataFrame with engineered features
    """
    print(f"\nLoading sliding windows from: {sliding_windows_file}")
    sliding_windows_df = pd.read_csv(sliding_windows_file)
    print(f"  Loaded {len(sliding_windows_df):,} rows")
    print(f"  Unique windows: {sliding_windows_df['sliding_window_id'].nunique()}")
    
    # Check class distribution
    if 'sliding_label' in sliding_windows_df.columns:
        class_dist = sliding_windows_df.groupby('sliding_window_id')['sliding_label'].first().value_counts()
        total_windows = sliding_windows_df['sliding_window_id'].nunique()
        print(f"\n  Class distribution:")
        print(f"    Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total_windows*100:.2f}%)")
        print(f"    Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total_windows*100:.2f}%)")
        if class_dist.get(1, 0) > 0:
            ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
            print(f"    Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    # Check for autoencoder features
    include_autoencoder = 'autoencoder_window_hits' in sliding_windows_df.columns
    print(f"\n  Autoencoder features available: {include_autoencoder}")
    
    # Group by window_id and engineer features
    print(f"\nEngineering features from {sliding_windows_df['sliding_window_id'].nunique()} windows...")
    
    feature_rows = []
    
    for window_id, window_data in tqdm(sliding_windows_df.groupby('sliding_window_id'),
                                       desc=f"Engineering {split_name} features",
                                       total=sliding_windows_df['sliding_window_id'].nunique()):
        try:
            # Engineer features
            features = engineer_features_for_window(
                window_data,
                global_mean=global_mean,
                global_std=global_std,
                include_autoencoder=include_autoencoder,
                window_size=window_size
            )
            
            # Add sliding window metadata
            features['sliding_window_id'] = window_id
            features['sliding_start_idx'] = window_data['sliding_start_idx'].iloc[0]
            features['sliding_end_idx'] = window_data['sliding_end_idx'].iloc[0]
            features['sliding_start_sclk'] = window_data['SCLK'].iloc[0]
            features['sliding_end_sclk'] = window_data['SCLK'].iloc[-1]
            
            # Add label from sliding windows
            if 'sliding_label' in window_data.columns:
                features['label'] = int(window_data['sliding_label'].iloc[0])
            
            feature_rows.append(features)
            
        except Exception as e:
            if verbose:
                print(f"  [WARNING] Failed to engineer features for window {window_id}: {e}")
            continue
    
    if not feature_rows:
        print("[ERROR] No features were engineered!")
        return None
    
    # Create features DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    print(f"\nFeature engineering complete!")
    print(f"  Total feature vectors: {len(features_df)}")
    print(f"  Features per vector: {len(features_df.columns) - 5}")  # Exclude metadata columns
    
    # Feature summary
    if 'label' in features_df.columns:
        class_dist = features_df['label'].value_counts()
        total = len(features_df)
        print(f"\n  Final class distribution:")
        print(f"    Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total*100:.2f}%)")
        print(f"    Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total*100:.2f}%)")
        if class_dist.get(1, 0) > 0:
            ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
            print(f"    Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    return features_df

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main feature engineering pipeline."""
    args = parse_arguments()
    
    print("=" * 70)
    print("FEATURE ENGINEERING FROM SLIDING WINDOWS")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Step size: {args.step_size}")
    print(f"Window size: {args.window_size}")
    
    # File paths
    sliding_windows_file = os.path.join(SLIDING_WINDOWS_DIR, 
                                       f"{args.split}_sliding_windows_step{args.step_size}.csv")
    
    if not os.path.exists(sliding_windows_file):
        print(f"[ERROR] Sliding windows file not found: {sliding_windows_file}")
        print("[INFO] Run generate_sliding_windows.py first!")
        return 1
    
    # Load training data to calculate global statistics
    print(f"\nLoading training data to calculate global statistics...")
    train_file = os.path.join(SPLITS_DIR, "ml_train.csv")
    
    if not os.path.exists(train_file):
        print(f"[ERROR] Training file not found: {train_file}")
        print("[INFO] Run data_preparation.py first!")
        return 1
    
    train_df = pd.read_csv(train_file)
    print(f"  Loaded {len(train_df):,} training samples")
    
    # Calculate global statistics (same as training)
    global_mean = float(train_df['PRESSURE'].mean())
    global_std = float(train_df['PRESSURE'].std())
    print(f"  Global mean: {global_mean:.2f} Pa")
    print(f"  Global std: {global_std:.2f} Pa")
    
    # Engineer features from sliding windows
    features_df = engineer_features_from_sliding_windows(
        sliding_windows_file,
        args.split,
        global_mean,
        global_std,
        args.window_size,
        args.step_size,
        args.verbose
    )
    
    if features_df is None:
        return 1
    
    # Save engineered features
    os.makedirs(FEATURES_DIR, exist_ok=True)
    output_file = os.path.join(FEATURES_DIR, f"{args.split}_sliding_features_step{args.step_size}.csv")
    
    print(f"\nSaving engineered features to: {output_file}")
    features_df.to_csv(output_file, index=False)
    
    print(f"  Saved {len(features_df):,} feature vectors")
    print(f"  Columns: {list(features_df.columns)}")
    
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING FROM SLIDING WINDOWS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Evaluate models on sliding windows")
    print(f"  2. Optimize decision thresholds")
    print(f"  3. Compare fixed vs sliding window performance")
    
    return 0

if __name__ == "__main__":
    exit(main())

