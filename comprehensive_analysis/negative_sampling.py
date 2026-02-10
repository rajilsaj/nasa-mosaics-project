#!/usr/bin/env python3
"""
Negative Sampling for Comprehensive Dataset
===========================================

As a seasoned RF scientist, this script:
1. Samples negative (non-vortex) windows from safe regions
2. Balances training set to 1:1 ratio (for effective learning)
3. Keeps validation/test with natural imbalance (for honest evaluation)
4. Preserves autoencoder features in negative windows

Strategy:
- Training: Balanced (1:1) - model learns from equal examples
- Validation/Test: Natural imbalance - realistic deployment scenario
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_SIZE = 60
SPLITS_DIR = "data/splits"
WINDOWS_DIR = "data/windows"
FEATURES_DIR = "data/features"
OUTPUT_DIR = "data/features"  # Save balanced features here

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate negative samples and create balanced datasets',
        epilog="""
Examples:
  python negative_sampling.py --split train --ratio 1.0
  python negative_sampling.py --split val --ratio 1.0  # For diagnostics
  python negative_sampling.py --split test --ratio 1.0  # For diagnostics
        """
    )
    
    parser.add_argument('--split', 
                       choices=['train', 'val', 'test'],
                       required=True,
                       help='Which temporal split to process')
    
    parser.add_argument('--ratio',
                       type=float,
                       default=1.0,
                       help='Negative to positive ratio (default: 1.0 for balanced)')
    
    parser.add_argument('--window_size',
                       type=int,
                       default=60,
                       help='Window size in samples (default: 60)')
    
    parser.add_argument('--buffer',
                       type=int,
                       default=50,
                       help='Buffer zone around positive windows (default: 50)')
    
    parser.add_argument('--random_seed',
                       type=int,
                       default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

# =============================================================================
# NEGATIVE SAMPLING FUNCTIONS
# =============================================================================

def create_forbidden_zones(ml_df, positive_windows_df, buffer):
    """
    Create forbidden zones where negative windows cannot be sampled.
    
    Forbidden zones include:
    1. All gt_detection_win == True regions (actual vortex regions)
    2. Buffer zones around positive windows (to avoid edge cases)
    
    Args:
        ml_df: ML dataset for this split (from data/splits/)
        positive_windows_df: Positive windows DataFrame (from data/windows/)
        buffer: Buffer size around positive windows (samples)
        
    Returns:
        Boolean array indicating forbidden indices
    """
    print(f"Creating forbidden zones (buffer={buffer} samples)...")
    
    forbidden = np.zeros(len(ml_df), dtype=bool)
    
    # Mark all gt_detection_win regions as forbidden
    forbidden[ml_df['gt_detection_win'] == True] = True
    
    # Mark buffer zones around positive windows
    for window_id in tqdm(positive_windows_df['window_id'].unique(), 
                         desc="Adding buffers", leave=False):
        window_data = positive_windows_df[positive_windows_df['window_id'] == window_id]
        
        # Find start and end SCLK values
        start_sclk = window_data['SCLK'].iloc[0]
        end_sclk = window_data['SCLK'].iloc[-1]
        
        # Find indices in ml_df
        start_matches = ml_df[ml_df['SCLK'] == start_sclk].index
        end_matches = ml_df[ml_df['SCLK'] == end_sclk].index
        
        if len(start_matches) > 0 and len(end_matches) > 0:
            start_idx = start_matches[0]
            end_idx = end_matches[0]
            
            # Add buffer
            buffer_start = max(0, start_idx - buffer)
            buffer_end = min(len(ml_df), end_idx + buffer)
            
            forbidden[buffer_start:buffer_end] = True
    
    forbidden_count = forbidden.sum()
    safe_count = (~forbidden).sum()
    
    print(f"  Forbidden zones: {forbidden_count:,} samples ({forbidden_count/len(ml_df)*100:.1f}%)")
    print(f"  Safe zones: {safe_count:,} samples ({safe_count/len(ml_df)*100:.1f}%)")
    
    return forbidden

def sample_negative_windows(ml_df, forbidden, num_samples, window_size, rng):
    """
    Sample negative windows from safe regions.
    
    Args:
        ml_df: ML dataset for this split
        forbidden: Boolean array of forbidden indices
        num_samples: Number of negative windows to sample
        window_size: Size of each window
        rng: Random number generator
        
    Returns:
        List of negative window DataFrames
    """
    print(f"Sampling {num_samples} negative windows...")
    
    # Find valid starting positions (where entire window is safe)
    valid_starts = []
    for i in range(len(ml_df) - window_size + 1):
        if not forbidden[i:i + window_size].any():
            valid_starts.append(i)
    
    print(f"  Found {len(valid_starts):,} valid starting positions")
    
    if len(valid_starts) < num_samples:
        print(f"  [WARNING] Only {len(valid_starts)} safe positions available")
        print(f"  [INFO] Reducing samples to {len(valid_starts)}")
        num_samples = len(valid_starts)
    
    # Sample starting positions (without replacement)
    sampled_starts = rng.choice(valid_starts, size=num_samples, replace=False)
    
    # Extract windows
    negative_windows = []
    for window_id, start_idx in enumerate(tqdm(sampled_starts, desc="Extracting negatives", leave=False)):
        window = ml_df.iloc[start_idx:start_idx + window_size].copy()
        
        # Add metadata (similar to positive windows)
        window['window_id'] = window_id + 100000  # Offset to avoid conflicts
        window['event_sclk'] = -1  # No associated vortex event
        window['split'] = ml_df.iloc[0].get('split', 'unknown') if 'split' in ml_df.columns else 'unknown'
        window['label'] = False  # Negative window
        
        negative_windows.append(window)
    
    return negative_windows

def engineer_features_for_negative_windows(negative_windows, global_mean, global_std, 
                                          include_autoencoder=False, window_size=60):
    """
    Engineer features for negative windows (same as positive windows).
    
    Args:
        negative_windows: List of negative window DataFrames
        global_mean: Global pressure mean
        global_std: Global pressure std
        include_autoencoder: Whether to include autoencoder features
        window_size: Window size
        
    Returns:
        List of feature dictionaries
    """
    # Import feature engineering function
    from feature_engineering import engineer_features_for_window
    
    all_features = []
    
    for window_data in tqdm(negative_windows, desc="Engineering negative features", leave=False):
        try:
            features = engineer_features_for_window(
                window_data,
                global_mean=global_mean,
                global_std=global_std,
                include_autoencoder=include_autoencoder,
                window_size=window_size
            )
            # Ensure label is False
            features['label'] = 0
            all_features.append(features)
        except Exception as e:
            print(f"  [WARNING] Failed to engineer features for negative window: {e}")
            continue
    
    return all_features

def combine_and_save(positive_features_df, negative_features_list, output_file):
    """
    Combine positive and negative features and save.
    
    Args:
        positive_features_df: DataFrame with positive window features
        negative_features_list: List of negative window feature dictionaries
        output_file: Output file path
    """
    print(f"\nCombining positive and negative features...")
    
    # Convert negative features to DataFrame
    if negative_features_list:
        negative_features_df = pd.DataFrame(negative_features_list)
    else:
        negative_features_df = pd.DataFrame()
    
    # Combine
    if len(negative_features_df) > 0:
        combined_df = pd.concat([positive_features_df, negative_features_df], 
                               ignore_index=True)
    else:
        combined_df = positive_features_df.copy()
    
    # Shuffle to mix positive and negative
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    print(f"Saving balanced features to: {output_file}")
    combined_df.to_csv(output_file, index=False)
    
    # Summary
    pos_count = (combined_df['label'] == 1).sum() if 'label' in combined_df.columns else 0
    neg_count = (combined_df['label'] == 0).sum() if 'label' in combined_df.columns else 0
    total = len(combined_df)
    
    print(f"\nBalanced dataset summary:")
    print(f"  Total windows: {total:,}")
    print(f"  Positive: {pos_count:,} ({pos_count/total*100:.1f}%)")
    print(f"  Negative: {neg_count:,} ({neg_count/total*100:.1f}%)")
    print(f"  Ratio: {neg_count/pos_count:.2f}:1 (Neg:Pos)" if pos_count > 0 else "  Ratio: N/A")
    print(f"  Total rows: {len(combined_df):,}")

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def main():
    """Main negative sampling pipeline."""
    args = parse_arguments()
    
    print("=" * 70)
    print(f"NEGATIVE SAMPLING - {args.split.upper()} SPLIT")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Split: {args.split}")
    print(f"  Ratio: {args.ratio}:1 (Neg:Pos)")
    print(f"  Window size: {args.window_size}")
    print(f"  Buffer: {args.buffer} samples")
    print(f"  Random seed: {args.random_seed}")
    
    # Set random seed
    rng = np.random.default_rng(args.random_seed)
    np.random.seed(args.random_seed)
    
    # File paths
    ml_file = os.path.join(SPLITS_DIR, f"ml_{args.split}.csv")
    positive_windows_file = os.path.join(WINDOWS_DIR, f"{args.split}_windows.csv")
    positive_features_file = os.path.join(FEATURES_DIR, f"{args.split}_features.csv")
    output_file = os.path.join(OUTPUT_DIR, f"{args.split}_balanced.csv")
    
    # Load data
    print(f"\nLoading data...")
    print(f"  ML dataset: {ml_file}")
    if not os.path.exists(ml_file):
        print(f"  [ERROR] File not found: {ml_file}")
        return 1
    ml_df = pd.read_csv(ml_file)
    print(f"    Loaded {len(ml_df):,} samples")
    
    print(f"  Positive windows: {positive_windows_file}")
    if not os.path.exists(positive_windows_file):
        print(f"  [ERROR] File not found: {positive_windows_file}")
        return 1
    positive_windows_df = pd.read_csv(positive_windows_file)
    num_positive = positive_windows_df['window_id'].nunique()
    print(f"    Loaded {num_positive} positive windows")
    
    print(f"  Positive features: {positive_features_file}")
    if not os.path.exists(positive_features_file):
        print(f"  [ERROR] Features not found. Run feature_engineering.py first!")
        return 1
    positive_features_df = pd.read_csv(positive_features_file)
    print(f"    Loaded {len(positive_features_df)} positive feature vectors")
    
    # Calculate number of negative samples
    num_negative = int(num_positive * args.ratio)
    print(f"\nSampling strategy:")
    print(f"  Positive windows: {num_positive}")
    print(f"  Negative windows: {num_negative} (ratio: {args.ratio}:1)")
    print(f"  Total windows: {num_positive + num_negative}")
    
    # Create forbidden zones
    forbidden = create_forbidden_zones(ml_df, positive_windows_df, args.buffer)
    
    # Sample negative windows
    negative_windows = sample_negative_windows(
        ml_df, forbidden, num_negative, args.window_size, rng
    )
    
    if len(negative_windows) == 0:
        print(f"\n[ERROR] No negative windows could be sampled!")
        return 1
    
    # Calculate global statistics for feature engineering
    print(f"\nCalculating global statistics for feature engineering...")
    global_mean = float(ml_df['PRESSURE'].mean())
    global_std = float(ml_df['PRESSURE'].std())
    print(f"  Global mean: {global_mean:.2f} Pa")
    print(f"  Global std: {global_std:.2f} Pa")
    
    # Check if autoencoder features should be included
    include_autoencoder = 'autoencoder_window_hits' in positive_windows_df.columns
    
    # Engineer features for negative windows
    print(f"\nEngineering features for negative windows...")
    negative_features_list = engineer_features_for_negative_windows(
        negative_windows,
        global_mean,
        global_std,
        include_autoencoder=include_autoencoder,
        window_size=args.window_size
    )
    
    if len(negative_features_list) == 0:
        print(f"\n[ERROR] No negative features were engineered!")
        return 1
    
    # Combine and save
    combine_and_save(positive_features_df, negative_features_list, output_file)
    
    print("\n" + "=" * 70)
    print("NEGATIVE SAMPLING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())

