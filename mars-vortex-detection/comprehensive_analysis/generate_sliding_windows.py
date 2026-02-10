#!/usr/bin/env python3
"""
Sliding Window Generator for Comprehensive Dataset
==================================================

Generates sliding windows from temporal splits with NASA's labeling logic
for realistic deployment simulation and evaluation.

Labeling Logic:
- True: Right-hand side of window falls within gt_detection_win (precursor region)
- False: Right-hand side comes before gt_detection_win
- Omit: Right-hand side is in gt_fwhm (actual vortex) or after

This simulates continuous monitoring deployment scenario.
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

SPLITS_DIR = "data/splits"
OUTPUT_DIR = "data/sliding_windows"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate sliding windows with NASA labeling logic',
        epilog="""
Examples:
  python generate_sliding_windows.py --split val --step_size 10
  python generate_sliding_windows.py --split test --step_size 10
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
    
    parser.add_argument('--step_size', 
                       type=int, 
                       default=10,
                       help='Step size for sliding window (default: 10)')
    
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

# =============================================================================
# LABELING LOGIC
# =============================================================================

def determine_window_label(window_data, ml_df_full):
    """
    Determine window label using NASA's specific logic.
    
    Args:
        window_data: DataFrame with window samples
        ml_df_full: Full ML dataset for this split
    
    Returns:
        str: 'True', 'False', or 'Omit'
    """
    if len(window_data) == 0:
        return 'Omit'
    
    # Get the right-hand side (end) of the window
    window_end_sclk = window_data['SCLK'].iloc[-1]
    
    # Find corresponding row in full ML dataset
    ml_row = ml_df_full[ml_df_full['SCLK'] == window_end_sclk]
    
    if len(ml_row) == 0:
        return 'Omit'
    
    ml_row = ml_row.iloc[0]
    
    # Apply NASA labeling logic
    if ml_row.get('gt_detection_win', False):
        return 'True'  # Right-hand side in precursor region
    elif ml_row.get('gt_fwhm', False):
        return 'Omit'  # Right-hand side in actual vortex (too late)
    else:
        return 'False'  # Right-hand side before precursor region

# =============================================================================
# SLIDING WINDOW GENERATION
# =============================================================================

def generate_sliding_windows(ml_df, split_name, window_size, step_size, verbose=False):
    """Generate sliding windows for a temporal split."""
    print(f"\nGenerating sliding windows for {split_name.upper()} split...")
    print(f"  ML samples: {len(ml_df):,}")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    expected_windows = (len(ml_df) - window_size) // step_size + 1
    print(f"  Expected windows: {expected_windows:,}")
    
    all_windows = []
    label_counts = {'True': 0, 'False': 0, 'Omit': 0}
    
    # Reset index for easier indexing
    ml_df_reset = ml_df.reset_index(drop=True)
    
    # Generate sliding windows
    for i in tqdm(range(0, len(ml_df_reset) - window_size + 1, step_size), 
                  desc=f"Generating {split_name} windows"):
        
        # Extract window
        window_data = ml_df_reset.iloc[i:i + window_size].copy()
        
        # Determine label using NASA logic
        label = determine_window_label(window_data, ml_df_reset)
        label_counts[label] += 1
        
        # Only keep True and False (omit 'Omit' windows)
        if label == 'Omit':
            continue
        
        # Store window (as DataFrame rows, not JSON)
        window_data = window_data.copy()
        window_data['sliding_window_id'] = len(all_windows)
        window_data['sliding_start_idx'] = i
        window_data['sliding_end_idx'] = i + window_size - 1
        window_data['sliding_label'] = 1 if label == 'True' else 0
        
        all_windows.append(window_data)
        
        if verbose and len(all_windows) % 10000 == 0:
            print(f"  Generated {len(all_windows):,} windows...")
    
    print(f"\n{split_name.upper()} sliding window generation complete!")
    print(f"  Total windows (True + False): {len(all_windows):,}")
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / (label_counts['True'] + label_counts['False'] + label_counts['Omit'])) * 100 if (label_counts['True'] + label_counts['False'] + label_counts['Omit']) > 0 else 0
        print(f"    {label}: {count:,} ({percentage:.1f}%)")
    
    # Count final distribution (after omitting 'Omit' windows)
    if all_windows:
        final_df = pd.concat(all_windows, ignore_index=True)
        final_pos = (final_df['sliding_label'] == 1).sum() if 'sliding_label' in final_df.columns else 0
        final_neg = (final_df['sliding_label'] == 0).sum() if 'sliding_label' in final_df.columns else 0
        print(f"\n  Final distribution (after omitting 'Omit' windows):")
        print(f"    Positive: {final_pos:,} ({final_pos/len(final_df)*100:.2f}%)")
        print(f"    Negative: {final_neg:,} ({final_neg/len(final_df)*100:.2f}%)")
        if final_pos > 0:
            print(f"    Ratio: {final_neg/final_pos:.1f}:1 (Neg:Pos)")
    
    return all_windows

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 70)
    print("SLIDING WINDOW GENERATION - COMPREHENSIVE DATASET")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Window size: {args.window_size}")
    print(f"Step size: {args.step_size}")
    print(f"Verbose: {args.verbose}")
    
    # File paths
    ml_file = os.path.join(SPLITS_DIR, f"ml_{args.split}.csv")
    
    if not os.path.exists(ml_file):
        print(f"[ERROR] ML split file not found: {ml_file}")
        print("[INFO] Run data_preparation.py first!")
        return 1
    
    # Load ML data
    print(f"\nLoading ML data: {ml_file}")
    ml_df = pd.read_csv(ml_file)
    print(f"  Loaded {len(ml_df):,} samples")
    
    # Ensure sorted by SCLK
    ml_df = ml_df.sort_values('SCLK').reset_index(drop=True)
    
    # Generate sliding windows
    windows = generate_sliding_windows(
        ml_df, args.split, args.window_size, args.step_size, args.verbose
    )
    
    if not windows:
        print("[WARNING] No windows generated!")
        return 1
    
    # Combine and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f"{args.split}_sliding_windows_step{args.step_size}.csv")
    
    print(f"\nCombining and saving windows...")
    all_windows_df = pd.concat(windows, ignore_index=True)
    all_windows_df.to_csv(output_file, index=False)
    
    print(f"  Saved {len(all_windows_df):,} rows to: {output_file}")
    print(f"  Unique windows: {all_windows_df['sliding_window_id'].nunique()}")
    
    # Check for autoencoder features
    if 'autoencoder_window_hits' in all_windows_df.columns:
        print(f"  Autoencoder features preserved: YES")
    
    print("\n" + "=" * 70)
    print("SLIDING WINDOW GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Engineer features from sliding windows")
    print(f"  2. Evaluate models on sliding windows")
    print(f"  3. Optimize thresholds for deployment")
    
    return 0

if __name__ == "__main__":
    exit(main())

