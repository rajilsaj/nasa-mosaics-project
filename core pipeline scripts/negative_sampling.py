"""
Negative Sampling for Mars Vortex Detection
============================================

This script generates negative (non-vortex) windows from safe regions
and combines them with positive windows to create balanced/natural datasets.

Strategy:
- Training: Balanced sampling (1:1 ratio for effective learning)
- Validation/Test: Natural sampling (realistic imbalance for honest evaluation)
"""

import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm


BASE_PATH = '/content/drive/MyDrive/2026/www/raw/'
# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate negative samples and combine with positive windows',
        epilog="""
Examples:
  python negative_sampling.py --split train --ratio 1.0
  python negative_sampling.py --split val --ratio 10.0
  python negative_sampling.py --split test --ratio 10.0
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
    1. All gt_detection_win == True regions
    2. Buffer zones around positive windows
    
    Args:
        ml_df: ML dataset for this split
        positive_windows_df: Positive windows DataFrame
        buffer: Buffer size around positive windows
        
    Returns:
        Boolean array indicating forbidden indices
    """
    print(f"Creating forbidden zones (buffer={buffer} samples)...")
    
    forbidden = np.zeros(len(ml_df), dtype=bool)
    
    # Mark all gt_detection_win regions as forbidden
    forbidden[ml_df['gt_detection_win'] == True] = True
    
    # Mark buffer zones around positive windows
    for window_id in tqdm(positive_windows_df['window_id'].unique(), desc="Adding buffers"):
        window_data = positive_windows_df[positive_windows_df['window_id'] == window_id]
        
        # Find start and end indices in ml_df
        start_sclk = window_data['SCLK'].iloc[0]
        end_sclk = window_data['SCLK'].iloc[-1]
        
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
        print(f"  WARNING: Only {len(valid_starts)} safe positions available")
        num_samples = len(valid_starts)
    
    # Sample starting positions
    sampled_starts = rng.choice(valid_starts, size=num_samples, replace=False)
    
    # Extract windows
    negative_windows = []
    for window_id, start_idx in enumerate(tqdm(sampled_starts, desc="Extracting negatives")):
        window = ml_df.iloc[start_idx:start_idx + window_size].copy()
        window['window_id'] = window_id
        window['event_sclk'] = -1  # No associated vortex event
        negative_windows.append(window)
    
    return negative_windows

def combine_and_save(positive_windows_df, negative_windows, output_file):
    """
    Combine positive and negative windows and save to file.
    
    Args:
        positive_windows_df: Positive windows DataFrame
        negative_windows: List of negative window DataFrames
        output_file: Output file path
    """
    print(f"Combining positive and negative windows...")
    
    # Convert positive windows to list format for consistency
    positive_list = []
    for window_id in positive_windows_df['window_id'].unique():
        window = positive_windows_df[positive_windows_df['window_id'] == window_id].copy()
        # Ensure label column exists
        if 'label' not in window.columns:
            window['label'] = 1  # Positive label
        positive_list.append(window)
    
    # Add label to negative windows
    for window in negative_windows:
        window['label'] = 0  # Negative label
    
    # Combine all windows
    all_windows = positive_list + negative_windows
    
    # Sort by starting SCLK to maintain temporal order
    all_windows.sort(key=lambda w: w['SCLK'].iloc[0])
    
    # Reassign window IDs
    for i, window in enumerate(all_windows):
        window['window_id'] = i
    
    # Concatenate into single DataFrame
    combined_df = pd.concat(all_windows, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    
    # Print summary
    class_counts = combined_df.groupby('window_id')['label'].first().value_counts()
    total_windows = len(all_windows)
    
    print(f"\n[SUCCESS] Saved combined dataset to: {output_file}")
    print(f"  Total windows: {total_windows}")
    print(f"  Positive windows: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/total_windows*100:.1f}%)")
    print(f"  Negative windows: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/total_windows*100:.1f}%)")
    print(f"  Total rows: {len(combined_df):,}")

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main negative sampling pipeline."""
    args = parse_arguments()
    
    print("="*70)
    print(f"NEGATIVE SAMPLING - {args.split.upper()} SPLIT")
    print("="*70)
    
    # Set random seed
    rng = np.random.default_rng(args.random_seed)
    np.random.seed(args.random_seed)
    
    # File paths
   

    ml_file = os.path.join(BASE_PATH, "temporal_splits", f"ml_{args.split}.csv")
    #ml_file = os.path.join("temporal_splits", f"ml_{args.split}.csv")
    positive_windows_file = f"{args.split}_windows.csv"
    output_file = f"{args.split}_balanced.csv"
    
    # Load data
    print(f"Loading ML dataset: {ml_file}")
    ml_df = pd.read_csv(ml_file)
    print(f"  Loaded {len(ml_df):,} samples")
    
    print(f"Loading positive windows: {positive_windows_file}")
    positive_windows_df = pd.read_csv(positive_windows_file)
    num_positive = positive_windows_df['window_id'].nunique()
    print(f"  Loaded {num_positive} positive windows")
    
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
    
    # Combine and save
    combine_and_save(positive_windows_df, negative_windows, output_file)
    
    print("\n" + "="*70)
    print("NEGATIVE SAMPLING COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
