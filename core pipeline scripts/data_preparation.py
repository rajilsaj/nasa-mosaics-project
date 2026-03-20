"""
Data Preparation with Temporal Splitting and Window Extraction for Mars Vortex Detection
=======================================================================================

This script implements the optimal architecture for time-series ML:
1. Load raw ML and Jackson datasets
2. Perform temporal splitting with gaps (train/val/test)
3. Extract windows from each split separately
4. Maintain causal integrity and prevent data leakage

Architecture:
Raw Data → Temporal Split → Extract Windows → Feature Engineering → Train

Benefits:
- Zero data leakage between splits
- Temporal isolation maintained
- Realistic evaluation scenarios
- Cleaner pipeline design
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

# Temporal split configuration
TRAIN_RATIO = 0.60      # 60% for training
GAP_RATIO = 0.005       # 0.5% gap between splits (~2 hours)
VAL_RATIO = 0.15        # 15% for validation
# Test gets remaining ~19.5%

# Window extraction parameters
WINDOW_SIZE = 60        # 60 samples backward from precursor region
NEGATIVE_SAMPLE_BUFFER = 50  # Buffer around positive events for negative sampling

# File paths
#ML_FILE = "ml_ready_vortex_data.csv"
#JACKSON_FILE = "Jackson_vortex_detections_reformatted_augmented.csv"
#OUTPUT_DIR = "temporal_splits"

BASE_PATH    = '/content/drive/MyDrive/2026/www/raw/'

ML_FILE      = os.path.join(BASE_PATH, "ml_ready_vortex_data.csv")
JACKSON_FILE = os.path.join(BASE_PATH, "Jackson_vortex_detections_reformatted_augmented.csv")

# Verify files exist before running anything
for f in [ML_FILE, JACKSON_FILE]:
    status = "✅ Found" if os.path.exists(f) else "❌ NOT FOUND"
    print(f"{status}: {f}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Temporal splitting and window extraction for Mars vortex detection',
        epilog="""
Examples:
  python data_preparation.py --extract_windows
  python data_preparation.py --extract_windows --window_size 75
  python data_preparation.py --split_only
        """
    )
    
    parser.add_argument('--extract_windows', action='store_true',
                       help='Extract windows after splitting')
    parser.add_argument('--split_only', action='store_true',
                       help='Only perform temporal splitting (no window extraction)')
    parser.add_argument('--window_size', type=int, default=60,
                       help='Window size for extraction (default: 60)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()

def load_and_validate_data():
    """Load and validate the raw datasets."""
    print("=" * 70)
    print("LOADING RAW DATASETS")
    print("=" * 70)
    
    # Load datasets
    print(f"Loading ML dataset: {ML_FILE}")
    ml_df = pd.read_csv(ML_FILE)
    print(f"  Loaded {len(ml_df):,} pressure samples")
    
    print(f"Loading Jackson dataset: {JACKSON_FILE}")
    jackson_df = pd.read_csv(JACKSON_FILE)
    print(f"  Loaded {len(jackson_df)} vortex events")
    
    # Cast SCLK to numeric
    print("\nValidating SCLK columns...")
    ml_df['SCLK'] = pd.to_numeric(ml_df['SCLK'], errors='coerce')
    jackson_df['SCLK'] = pd.to_numeric(jackson_df['SCLK'], errors='coerce')
    
    # Check for conversion failures
    ml_nan_count = ml_df['SCLK'].isna().sum()
    jackson_nan_count = jackson_df['SCLK'].isna().sum()
    
    if ml_nan_count > 0:
        print(f"WARNING: {ml_nan_count} invalid SCLK values in ML dataset")
        ml_df = ml_df.dropna(subset=['SCLK'])
    
    if jackson_nan_count > 0:
        print(f"WARNING: {jackson_nan_count} invalid SCLK values in Jackson dataset")
        jackson_df = jackson_df.dropna(subset=['SCLK'])
    
    # Sort by SCLK to ensure temporal order
    ml_df = ml_df.sort_values('SCLK').reset_index(drop=True)
    jackson_df = jackson_df.sort_values('SCLK').reset_index(drop=True)
    
    print(f"\nData validation:")
    print(f"  ML SCLK range: {ml_df['SCLK'].min()} to {ml_df['SCLK'].max()}")
    print(f"  Jackson SCLK range: {jackson_df['SCLK'].min()} to {jackson_df['SCLK'].max()}")
    print(f"  ML samples: {len(ml_df):,}")
    print(f"  Jackson events: {len(jackson_df):,}")
    
    return ml_df, jackson_df

def perform_temporal_splitting(ml_df, jackson_df):
    """Perform temporal splitting with gaps to prevent data leakage."""
    print("\n" + "=" * 70)
    print("TEMPORAL SPLITTING WITH GAPS")
    print("=" * 70)
    
    n = len(ml_df)
    
    # Calculate split indices with gaps
    train_end_idx = int(n * TRAIN_RATIO)
    val_start_idx = int(n * (TRAIN_RATIO + GAP_RATIO))
    val_end_idx = int(n * (TRAIN_RATIO + GAP_RATIO + VAL_RATIO))
    test_start_idx = int(n * (TRAIN_RATIO + 2 * GAP_RATIO + VAL_RATIO))
    
    print(f"Split configuration:")
    print(f"  Training: 0 to {train_end_idx:,} ({TRAIN_RATIO*100:.1f}%)")
    print(f"  Gap 1: {train_end_idx:,} to {val_start_idx:,} ({GAP_RATIO*100:.1f}%)")
    print(f"  Validation: {val_start_idx:,} to {val_end_idx:,} ({VAL_RATIO*100:.1f}%)")
    print(f"  Gap 2: {val_end_idx:,} to {test_start_idx:,} ({GAP_RATIO*100:.1f}%)")
    print(f"  Test: {test_start_idx:,} to {n:,} ({(1-TRAIN_RATIO-2*GAP_RATIO-VAL_RATIO)*100:.1f}%)")
    
    # Split ML data
    ml_train = ml_df.iloc[:train_end_idx].copy()
    ml_val = ml_df.iloc[val_start_idx:val_end_idx].copy()
    ml_test = ml_df.iloc[test_start_idx:].copy()
    
    print(f"\nML split sizes:")
    print(f"  Train: {len(ml_train):,} samples")
    print(f"  Validation: {len(ml_val):,} samples")
    print(f"  Test: {len(ml_test):,} samples")
    
    # Split Jackson data based on SCLK ranges
    train_sclk_max = ml_train['SCLK'].max()
    val_sclk_min = ml_val['SCLK'].min()
    val_sclk_max = ml_val['SCLK'].max()
    test_sclk_min = ml_test['SCLK'].min()
    
    jackson_train = jackson_df[jackson_df['SCLK'] <= train_sclk_max].copy()
    jackson_val = jackson_df[(jackson_df['SCLK'] >= val_sclk_min) & 
                            (jackson_df['SCLK'] <= val_sclk_max)].copy()
    jackson_test = jackson_df[jackson_df['SCLK'] >= test_sclk_min].copy()
    
    print(f"\nJackson split sizes:")
    print(f"  Train: {len(jackson_train)} events")
    print(f"  Validation: {len(jackson_val)} events")
    print(f"  Test: {len(jackson_test)} events")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save splits
    print(f"\nSaving temporal splits to {OUTPUT_DIR}/...")
    ml_train.to_csv(os.path.join(OUTPUT_DIR, "ml_train.csv"), index=False)
    ml_val.to_csv(os.path.join(OUTPUT_DIR, "ml_val.csv"), index=False)
    ml_test.to_csv(os.path.join(OUTPUT_DIR, "ml_test.csv"), index=False)
    
    jackson_train.to_csv(os.path.join(OUTPUT_DIR, "jackson_train.csv"), index=False)
    jackson_val.to_csv(os.path.join(OUTPUT_DIR, "jackson_val.csv"), index=False)
    jackson_test.to_csv(os.path.join(OUTPUT_DIR, "jackson_test.csv"), index=False)
    
    print("  [SUCCESS] All temporal splits saved!")
    
    return {
        'ml': {'train': ml_train, 'val': ml_val, 'test': ml_test},
        'jackson': {'train': jackson_train, 'val': jackson_val, 'test': jackson_test}
    }

def extract_windows_from_split(ml_df, jackson_df, split_name, window_size, verbose=False):
    """Extract windows from a single temporal split using position-based indexing."""
    print(f"\nExtracting {window_size}-step windows from {split_name.upper()} split...")
    print(f"  ML samples: {len(ml_df):,}")
    print(f"  Jackson events: {len(jackson_df)}")
    
    if verbose:
        print(f"  DEBUG: ml_df type: {type(ml_df)}")
        print(f"  DEBUG: ml_df index range: {ml_df.index.min()} to {ml_df.index.max()}")
        print(f"  DEBUG: ml_df is empty: {ml_df.empty}")
    
    all_windows = []
    successful_count = 0
    failed_count = 0
    
    # Reset index to ensure we work with 0-based positions
    ml_df_reset = ml_df.reset_index(drop=True)
    
    for idx, row in tqdm(jackson_df.iterrows(), total=len(jackson_df), 
                        desc=f"Processing {split_name} events"):
        target_sclk = row['SCLK']
        
        # Find the SCLK in ML dataset using position-based search
        matches = ml_df_reset[ml_df_reset['SCLK'] == target_sclk]
        if matches.empty:
            if verbose:
                print(f"  WARNING: SCLK {target_sclk} not found in {split_name} ML data")
            failed_count += 1
            continue
        
        # Get position in the reset dataframe (0-based)
        target_position = matches.index[0]
        
        if verbose:
            print(f"    DEBUG: SCLK {target_sclk} found at position {target_position}")
        
        # Look backward to find where gt_detection_win == True
        # Search from beginning to target position
        search_space = ml_df_reset.iloc[:target_position + 1]  # Include target position
        precursor_region = search_space[search_space['gt_detection_win'] == True]
        
        if precursor_region.empty:
            if verbose:
                print(f"  WARNING: No gt_detection_win=True found for SCLK {target_sclk}")
            failed_count += 1
            continue
        
        # Get the position of the FIRST gt_detection_win=True (where precursor starts)
        first_precursor_position = precursor_region.index[0]
        
        # Go back 60 steps from the FIRST precursor position (before precursor starts)
        start_position = max(0, first_precursor_position - window_size)
        end_position = first_precursor_position - 1  # Window ends just before precursor starts
        
        if verbose:
            print(f"    DEBUG: start_position={start_position}, end_position={end_position}")
            print(f"    DEBUG: Window would span {end_position - start_position + 1} samples")
        
        # Extract the window using position-based indexing
        try:
            window = ml_df_reset.iloc[start_position:end_position + 1].copy()
            
            if verbose:
                print(f"    DEBUG: Window extracted successfully, size={len(window)}")
            
            # Validate window size
            if len(window) < window_size:
                if verbose:
                    print(f"  WARNING: Window too small for SCLK {target_sclk}: {len(window)} < {window_size}")
                failed_count += 1
                continue
            
            # Add metadata
            window['window_id'] = successful_count
            window['event_sclk'] = target_sclk
            window['split'] = split_name
            
            all_windows.append(window)
            successful_count += 1
            
            if verbose:
                print(f"  DEBUG: SCLK {target_sclk} - Window size: {len(window)}, Expected: {window_size}")
                
        except Exception as e:
            if verbose:
                print(f"  ERROR: Failed to extract window for SCLK {target_sclk}: {e}")
            failed_count += 1
            continue
    
    print(f"\n{split_name.upper()} window extraction results:")
    print(f"  Successfully extracted: {successful_count} windows")
    print(f"  Failed: {failed_count} windows")
    if successful_count + failed_count > 0:
        print(f"  Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
    else:
        print(f"  Success rate: 0.0%")
    
    return all_windows

def extract_all_windows(splits, window_size, verbose=False):
    """Extract windows from all temporal splits."""
    print("\n" + "=" * 70)
    print("WINDOW EXTRACTION FROM TEMPORAL SPLITS")
    print("=" * 70)
    
    all_split_windows = {}
    
    for split_name in ['train', 'val', 'test']:
        ml_df = splits['ml'][split_name]
        jackson_df = splits['jackson'][split_name]
        
        windows = extract_windows_from_split(ml_df, jackson_df, split_name, window_size, verbose)
        all_split_windows[split_name] = windows
    
    # Save windows for each split
    print(f"\nSaving extracted windows...")
    for split_name, windows in all_split_windows.items():
        if windows:
            result_df = pd.concat(windows, ignore_index=True)
            output_file = f"{split_name}_windows.csv"
            result_df.to_csv(output_file, index=False)
            
            print(f"  {split_name.upper()}: {len(result_df):,} rows ({len(windows)} windows) -> {output_file}")
            
            # Coverage analysis
            original_gt_count = splits['ml'][split_name]['gt_detection_win'].sum()
            extracted_gt_count = result_df['gt_detection_win'].sum()
            coverage = (extracted_gt_count / original_gt_count) * 100 if original_gt_count > 0 else 0
            
            print(f"    Coverage: {coverage:.2f}% ({extracted_gt_count:,}/{original_gt_count:,})")
        else:
            print(f"  {split_name.upper()}: No windows extracted")
    
    return all_split_windows

def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 70)
    print("DATA PREPARATION - TEMPORAL SPLITTING & WINDOW EXTRACTION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Extract windows: {args.extract_windows}")
    print(f"  Split only: {args.split_only}")
    print(f"  Window size: {args.window_size}")
    print(f"  Verbose: {args.verbose}")
    
    try:
        # Step 1: Load and validate data
        ml_df, jackson_df = load_and_validate_data()
        
        # Step 2: Perform temporal splitting
        splits = perform_temporal_splitting(ml_df, jackson_df)
        
        # Step 3: Extract windows (if requested)
        if args.extract_windows:
            all_windows = extract_all_windows(splits, args.window_size, args.verbose)
        
        print("\n" + "=" * 70)
        print("DATA PREPARATION COMPLETED SUCCESSFULLY")
        print("=" * 70)
        
        if args.extract_windows:
            total_windows = sum(len(windows) for windows in all_windows.values())
            print(f"Total windows extracted: {total_windows}")
            print(f"Ready for feature engineering and training!")
        else:
            print(f"Temporal splits ready in {OUTPUT_DIR}/")
            print(f"Run with --extract_windows to extract windows from splits")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())