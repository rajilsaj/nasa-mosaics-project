#!/usr/bin/env python3
"""
Sliding Window Generator for Mars Vortex Detection
=================================================

This script generates sliding windows from temporal splits with NASA's specific
labeling logic for realistic deployment simulation and evaluation.

Labeling Logic:
- True: Right-hand side of window falls within gt_detection_win (precursor region)
- False: Right-hand side comes before gt_detection_win
- Omit: Right-hand side is in gt_fwhm (actual vortex) or after
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate sliding windows with NASA labeling logic',
        epilog="""
Examples:
  python sliding_window_generator.py --split train --step_size 10
  python sliding_window_generator.py --split val --step_size 5
  python sliding_window_generator.py --split test --step_size 10
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

def determine_window_label(window_data, jackson_events_df, ml_df_full):
    """
    Determine window label using NASA's specific logic.
    
    Args:
        window_data: DataFrame with window samples
        jackson_events_df: DataFrame with Jackson vortex events for this split
        ml_df_full: Full ML dataset for this split
    
    Returns:
        str: 'True', 'False', or 'Omit'
    """
    if len(window_data) == 0:
        return 'Omit'
    
    # Get the right-hand side (end) of the window
    window_end_sclk = window_data['SCLK'].iloc[-1]
    window_end_idx = window_data.index[-1]
    
    # Find corresponding row in full ML dataset
    ml_row = ml_df_full.iloc[window_end_idx]
    
    # Apply NASA labeling logic
    if ml_row['gt_detection_win']:
        return 'True'  # Right-hand side in precursor region
    elif ml_row['gt_fwhm']:
        return 'Omit'  # Right-hand side in actual vortex (too late)
    else:
        return 'False'  # Right-hand side before precursor region

def generate_sliding_windows(ml_df, jackson_df, split_name, window_size, step_size, verbose=False):
    """Generate sliding windows for a temporal split."""
    print(f"\nGenerating sliding windows for {split_name.upper()} split...")
    print(f"  ML samples: {len(ml_df):,}")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    print(f"  Expected windows: {(len(ml_df) - window_size) // step_size + 1:,}")
    
    all_windows = []
    label_counts = {'True': 0, 'False': 0, 'Omit': 0}
    
    # Generate sliding windows
    for i in tqdm(range(0, len(ml_df) - window_size + 1, step_size), 
                  desc=f"Generating {split_name} windows"):
        
        # Extract window
        window_data = ml_df.iloc[i:i + window_size].copy()
        
        # Determine label using NASA logic
        label = determine_window_label(window_data, jackson_df, ml_df)
        label_counts[label] += 1
        
        # Store window metadata
        window_info = {
            'window_id': len(all_windows),
            'start_idx': i,
            'end_idx': i + window_size - 1,
            'start_sclk': window_data['SCLK'].iloc[0],
            'end_sclk': window_data['SCLK'].iloc[-1],
            'label': label,
            'window_data': window_data.to_json(orient='records')
        }
        
        all_windows.append(window_info)
        
        if verbose and len(all_windows) % 10000 == 0:
            print(f"  Generated {len(all_windows):,} windows...")
    
    print(f"\n{split_name.upper()} sliding window generation complete!")
    print(f"  Total windows: {len(all_windows):,}")
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(all_windows)) * 100
        print(f"    {label}: {count:,} ({percentage:.1f}%)")
    
    return all_windows

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 70)
    print("SLIDING WINDOW GENERATION - MARS VORTEX DETECTION")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Window size: {args.window_size}")
    print(f"Step size: {args.step_size}")
    print(f"Verbose: {args.verbose}")
    
    # File paths
    ml_file = f"temporal_splits/ml_{args.split}.csv"
    jackson_file = f"temporal_splits/jackson_{args.split}.csv"
    output_file = f"{args.split}_sliding_windows_step{args.step_size}.csv"
    
    try:
        # Load data
        print(f"\nLoading data...")
        ml_df = pd.read_csv(ml_file)
        jackson_df = pd.read_csv(jackson_file)
        
        print(f"  ML data: {len(ml_df):,} samples")
        print(f"  Jackson data: {len(jackson_df)} events")
        
        # Generate sliding windows
        windows = generate_sliding_windows(
            ml_df, jackson_df, args.split, 
            args.window_size, args.step_size, args.verbose
        )
        
        # Save to CSV
        windows_df = pd.DataFrame(windows)
        windows_df.to_csv(output_file, index=False)
        
        print(f"\n[SUCCESS] Saved {len(windows):,} sliding windows to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        print("\n" + "=" * 70)
        print("SLIDING WINDOW GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
