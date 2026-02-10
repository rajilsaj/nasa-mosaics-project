"""
Temporal Data Splitting for Mars Vortex Detection
==================================================

This script creates temporally-isolated train/validation/test splits to maintain
causality and prevent data leakage. Splits are based on chronological order (SCLK)
with gaps between splits to ensure temporal isolation.

Key Principles:
- Train on early data, test on later data (maintains causality)
- Temporal gaps between splits (prevents leakage)
- Realistic evaluation (model learns from past, predicts future)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

ML_FILE = "ml_ready_vortex_data.csv"
JACKSON_FILE = "Jackson_vortex_detections_reformatted_augmented.csv"
OUTPUT_DIR = "temporal_splits"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
GAP_RATIO = 0.035  # 3.5% gap between splits

# =============================================================================
# TEMPORAL SPLITTING FUNCTIONS
# =============================================================================

def create_temporal_splits(ml_df, jackson_df):
    """
    Create temporally-isolated train/validation/test splits.
    
    Strategy:
    - Train: First 70% of data
    - Gap: 3.5% buffer
    - Val: Next 15% of data
    - Gap: 3.5% buffer  
    - Test: Final 15% of data
    
    Args:
        ml_df: Full ML dataset sorted by SCLK
        jackson_df: Jackson vortex events sorted by SCLK
        
    Returns:
        Dictionary with train/val/test splits for both datasets
    """
    print("Creating temporal splits with gaps...")
    
    n = len(ml_df)
    
    # Calculate split indices
    train_end = int(n * TRAIN_RATIO)
    gap1_end = int(n * (TRAIN_RATIO + GAP_RATIO))
    val_start = gap1_end
    val_end = int(n * (TRAIN_RATIO + GAP_RATIO + VAL_RATIO))
    gap2_end = int(n * (TRAIN_RATIO + 2*GAP_RATIO + VAL_RATIO))
    test_start = gap2_end
    
    # Create ML dataset splits
    ml_splits = {
        'train': ml_df.iloc[:train_end].reset_index(drop=True),
        'val': ml_df.iloc[val_start:val_end].reset_index(drop=True),
        'test': ml_df.iloc[test_start:].reset_index(drop=True)
    }
    
    # Create Jackson dataset splits (filter by SCLK ranges)
    jackson_splits = {}
    
    for split_name, split_df in ml_splits.items():
        min_sclk = split_df['SCLK'].min()
        max_sclk = split_df['SCLK'].max()
        
        jackson_subset = jackson_df[
            (jackson_df['SCLK'] >= min_sclk) & 
            (jackson_df['SCLK'] <= max_sclk)
        ].reset_index(drop=True)
        
        jackson_splits[split_name] = jackson_subset
    
    return ml_splits, jackson_splits

def validate_temporal_isolation(ml_splits):
    """
    Validate that temporal splits maintain isolation (no overlap).
    
    Args:
        ml_splits: Dictionary with train/val/test ML splits
        
    Returns:
        Boolean indicating if splits are temporally isolated
    """
    print("\nValidating temporal isolation...")
    
    train_max = ml_splits['train']['SCLK'].max()
    val_min = ml_splits['val']['SCLK'].min()
    val_max = ml_splits['val']['SCLK'].max()
    test_min = ml_splits['test']['SCLK'].min()
    
    # Check for temporal overlap
    train_val_overlap = train_max >= val_min
    val_test_overlap = val_max >= test_min
    
    print(f"  Train SCLK range: {ml_splits['train']['SCLK'].min()} to {train_max}")
    print(f"  Val SCLK range: {val_min} to {val_max}")
    print(f"  Test SCLK range: {test_min} to {ml_splits['test']['SCLK'].max()}")
    
    # Calculate gaps
    gap1_hours = (val_min - train_max) / 3600 if not train_val_overlap else 0
    gap2_hours = (test_min - val_max) / 3600 if not val_test_overlap else 0
    
    print(f"  Train-Val gap: {gap1_hours:.1f} hours")
    print(f"  Val-Test gap: {gap2_hours:.1f} hours")
    
    # Validation results
    if not train_val_overlap and not val_test_overlap:
        print("  ✅ TEMPORAL ISOLATION CONFIRMED")
        return True
    else:
        print("  ❌ TEMPORAL OVERLAP DETECTED")
        if train_val_overlap:
            print(f"    Train-Val overlap: {train_max - val_min} seconds")
        if val_test_overlap:
            print(f"    Val-Test overlap: {val_max - test_min} seconds")
        return False

def save_temporal_splits(ml_splits, jackson_splits, output_dir):
    """
    Save temporal splits to CSV files.
    
    Args:
        ml_splits: Dictionary with train/val/test ML splits
        jackson_splits: Dictionary with train/val/test Jackson splits
        output_dir: Output directory for split files
    """
    print(f"\nSaving temporal splits to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ML dataset splits
    for split_name in ['train', 'val', 'test']:
        ml_file = os.path.join(output_dir, f"ml_{split_name}.csv")
        ml_splits[split_name].to_csv(ml_file, index=False)
        print(f"  Saved ML {split_name}: {len(ml_splits[split_name]):,} samples")
    
    # Save Jackson dataset splits
    for split_name in ['train', 'val', 'test']:
        jackson_file = os.path.join(output_dir, f"jackson_{split_name}.csv")
        jackson_splits[split_name].to_csv(jackson_file, index=False)
        print(f"  Saved Jackson {split_name}: {len(jackson_splits[split_name])} events")
    
    # Save split summary
    summary_file = os.path.join(output_dir, "split_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Temporal Split Summary\n")
        f.write("=====================\n\n")
        f.write(f"Split ratios: Train={TRAIN_RATIO:.1%}, Val={VAL_RATIO:.1%}, Test={TEST_RATIO:.1%}\n")
        f.write(f"Gap ratio: {GAP_RATIO:.1%}\n\n")
        
        for split_name in ['train', 'val', 'test']:
            ml_df = ml_splits[split_name]
            jackson_df = jackson_splits[split_name]
            
            f.write(f"{split_name.upper()} Split:\n")
            f.write(f"  ML samples: {len(ml_df):,}\n")
            f.write(f"  SCLK range: {ml_df['SCLK'].min()} to {ml_df['SCLK'].max()}\n")
            f.write(f"  Jackson events: {len(jackson_df)}\n")
            f.write(f"  Vortex events: {len(jackson_df)} events\n\n")

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main temporal splitting pipeline."""
    print("="*70)
    print("MARS VORTEX DETECTION - TEMPORAL SPLITTING")
    print("="*70)
    
    # Load datasets
    print(f"Loading ML dataset: {ML_FILE}")
    ml_df = pd.read_csv(ML_FILE)
    print(f"  Loaded {len(ml_df):,} pressure samples")
    
    print(f"Loading Jackson dataset: {JACKSON_FILE}")
    jackson_df = pd.read_csv(JACKSON_FILE)
    print(f"  Loaded {len(jackson_df)} vortex events")
    
    # Cast SCLK to numeric
    ml_df['SCLK'] = pd.to_numeric(ml_df['SCLK'], errors='coerce')
    jackson_df['SCLK'] = pd.to_numeric(jackson_df['SCLK'], errors='coerce')
    
    # Sort by SCLK to ensure temporal order
    ml_df = ml_df.sort_values('SCLK').reset_index(drop=True)
    jackson_df = jackson_df.sort_values('SCLK').reset_index(drop=True)
    
    print(f"\nSCLK ranges:")
    print(f"  ML dataset: {ml_df['SCLK'].min()} to {ml_df['SCLK'].max()}")
    print(f"  Jackson dataset: {jackson_df['SCLK'].min()} to {jackson_df['SCLK'].max()}")
    
    # Create temporal splits
    ml_splits, jackson_splits = create_temporal_splits(ml_df, jackson_df)
    
    # Validate temporal isolation
    is_valid = validate_temporal_isolation(ml_splits)
    
    if not is_valid:
        print("\n❌ Temporal splitting failed validation!")
        return
    
    # Save splits
    save_temporal_splits(ml_splits, jackson_splits, OUTPUT_DIR)
    
    print(f"\n" + "="*70)
    print("TEMPORAL SPLITTING COMPLETED SUCCESSFULLY")
    print("="*70)
    
    print(f"\nNext steps:")
    print(f"  1. Use ml_train.csv, ml_val.csv, ml_test.csv for window extraction")
    print(f"  2. Use jackson_train.csv, jackson_val.csv, jackson_test.csv for events")
    print(f"  3. Apply balanced sampling to training set")
    print(f"  4. Apply natural sampling to validation/test sets")

if __name__ == "__main__":
    main()
