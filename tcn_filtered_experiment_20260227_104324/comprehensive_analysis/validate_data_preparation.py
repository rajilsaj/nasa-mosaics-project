#!/usr/bin/env python3
"""
Validate Data Preparation Results
=================================
Validates temporal splits and window extraction for comprehensive dataset.
Checks for data quality, temporal isolation, and autoencoder feature preservation.
"""

import pandas as pd
import numpy as np
import os

def validate_temporal_splits():
    """Validate temporal splits are correct."""
    print("=" * 70)
    print("VALIDATING TEMPORAL SPLITS")
    print("=" * 70)
    
    splits_dir = "data/splits"
    if not os.path.exists(splits_dir):
        print(f"[ERROR] {splits_dir} does not exist!")
        return False
    
    # Load splits
    try:
        ml_train = pd.read_csv(os.path.join(splits_dir, "ml_train.csv"))
        ml_val = pd.read_csv(os.path.join(splits_dir, "ml_val.csv"))
        ml_test = pd.read_csv(os.path.join(splits_dir, "ml_test.csv"))
        
        jackson_train = pd.read_csv(os.path.join(splits_dir, "jackson_train.csv"))
        jackson_val = pd.read_csv(os.path.join(splits_dir, "jackson_val.csv"))
        jackson_test = pd.read_csv(os.path.join(splits_dir, "jackson_test.csv"))
    except FileNotFoundError as e:
        print(f"[ERROR] Missing split file: {e}")
        return False
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(ml_train):,} samples, {len(jackson_train)} events")
    print(f"  Val:   {len(ml_val):,} samples, {len(jackson_val)} events")
    print(f"  Test:  {len(ml_test):,} samples, {len(jackson_test)} events")
    
    # Check temporal isolation
    print(f"\nTemporal isolation check:")
    train_max = ml_train['SCLK'].max()
    val_min = ml_val['SCLK'].min()
    val_max = ml_val['SCLK'].max()
    test_min = ml_test['SCLK'].min()
    
    gap1 = val_min - train_max
    gap2 = test_min - val_max
    
    print(f"  Train SCLK: {ml_train['SCLK'].min()} to {train_max}")
    print(f"  Gap 1: {gap1:,} SCLK units")
    print(f"  Val SCLK: {val_min} to {val_max}")
    print(f"  Gap 2: {gap2:,} SCLK units")
    print(f"  Test SCLK: {test_min} to {ml_test['SCLK'].max()}")
    
    if gap1 > 0 and gap2 > 0:
        print(f"  [OK] Temporal gaps present - no overlap")
    else:
        print(f"  [WARNING] Gaps may be too small or missing!")
        return False
    
    # Check for overlap
    train_sclks = set(ml_train['SCLK'])
    val_sclks = set(ml_val['SCLK'])
    test_sclks = set(ml_test['SCLK'])
    
    overlap_train_val = len(train_sclks & val_sclks)
    overlap_val_test = len(val_sclks & test_sclks)
    overlap_train_test = len(train_sclks & test_sclks)
    
    if overlap_train_val == 0 and overlap_val_test == 0 and overlap_train_test == 0:
        print(f"  [OK] No SCLK overlap between splits")
    else:
        print(f"  [ERROR] Overlap detected!")
        print(f"    Train-Val overlap: {overlap_train_val}")
        print(f"    Val-Test overlap: {overlap_val_test}")
        print(f"    Train-Test overlap: {overlap_train_test}")
        return False
    
    # Check class distribution
    print(f"\nClass distribution:")
    for split_name, df in [('Train', ml_train), ('Val', ml_val), ('Test', ml_test)]:
        pos = df['gt_detection_win'].sum()
        neg = (~df['gt_detection_win']).sum()
        total = len(df)
        print(f"  {split_name}: {pos:,} positive ({pos/total*100:.3f}%), {neg:,} negative ({neg/total*100:.3f}%)")
    
    # Check for autoencoder features
    print(f"\nAutoencoder features check:")
    has_ae = 'autoencoder_window_hits' in ml_train.columns
    print(f"  Autoencoder features present: {has_ae}")
    if has_ae:
        print(f"  [OK] Autoencoder features preserved in splits")
    else:
        print(f"  [WARNING] Autoencoder features not found (may be expected if not in source)")
    
    return True

def validate_windows():
    """Validate extracted windows."""
    print("\n" + "=" * 70)
    print("VALIDATING EXTRACTED WINDOWS")
    print("=" * 70)
    
    windows_dir = "data/windows"
    if not os.path.exists(windows_dir):
        print(f"[ERROR] {windows_dir} does not exist!")
        return False
    
    # Load windows
    try:
        train_windows = pd.read_csv(os.path.join(windows_dir, "train_windows.csv"))
        val_windows = pd.read_csv(os.path.join(windows_dir, "val_windows.csv"))
        test_windows = pd.read_csv(os.path.join(windows_dir, "test_windows.csv"))
    except FileNotFoundError as e:
        print(f"[ERROR] Missing window file: {e}")
        return False
    
    print(f"\nWindow counts:")
    train_window_count = train_windows['window_id'].nunique()
    val_window_count = val_windows['window_id'].nunique()
    test_window_count = test_windows['window_id'].nunique()
    
    print(f"  Train: {train_window_count} windows, {len(train_windows):,} rows")
    print(f"  Val:   {val_window_count} windows, {len(val_windows):,} rows")
    print(f"  Test:  {test_window_count} windows, {len(test_windows):,} rows")
    
    # Check window size
    print(f"\nWindow size validation:")
    for split_name, df in [('Train', train_windows), ('Val', val_windows), ('Test', test_windows)]:
        window_sizes = df.groupby('window_id').size()
        expected_size = 60
        correct_size = (window_sizes == expected_size).sum()
        total_windows = len(window_sizes)
        
        print(f"  {split_name}: {correct_size}/{total_windows} windows have correct size ({expected_size})")
        if correct_size < total_windows:
            print(f"    [WARNING] Some windows have incorrect size!")
            print(f"    Size range: {window_sizes.min()} to {window_sizes.max()}")
    
    # Check for autoencoder features in windows
    print(f"\nAutoencoder features in windows:")
    has_ae = 'autoencoder_window_hits' in train_windows.columns
    print(f"  Autoencoder features present: {has_ae}")
    if has_ae:
        print(f"  [OK] Autoencoder features preserved in windows")
        # Check distribution
        print(f"  autoencoder_window_hits range: {train_windows['autoencoder_window_hits'].min()} to {train_windows['autoencoder_window_hits'].max()}")
        ae_pos_count = train_windows['autoencoder_positive_hit'].sum()
        print(f"  autoencoder_positive_hit: {ae_pos_count:,} positive hits in train windows")
    else:
        print(f"  [WARNING] Autoencoder features not found in windows")
    
    # Check labels
    print(f"\nWindow labels:")
    for split_name, df in [('Train', train_windows), ('Val', val_windows), ('Test', test_windows)]:
        if 'label' in df.columns:
            pos_windows = (df.groupby('window_id')['label'].first() == True).sum()
            total_windows = df['window_id'].nunique()
            print(f"  {split_name}: {pos_windows}/{total_windows} positive windows ({pos_windows/total_windows*100:.1f}%)")
        else:
            print(f"  {split_name}: No 'label' column found")
    
    return True

def validate_data_quality():
    """Validate data quality metrics."""
    print("\n" + "=" * 70)
    print("VALIDATING DATA QUALITY")
    print("=" * 70)
    
    splits_dir = "data/splits"
    
    # Check each split
    for split_name in ['train', 'val', 'test']:
        file_path = os.path.join(splits_dir, f"ml_{split_name}.csv")
        if not os.path.exists(file_path):
            continue
        
        df = pd.read_csv(file_path)
        
        print(f"\n{split_name.upper()} split quality:")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"  [WARNING] Missing values found:")
            for col, count in missing[missing > 0].items():
                print(f"    {col}: {count:,} ({count/len(df)*100:.2f}%)")
        else:
            print(f"  [OK] No missing values")
        
        # Check SCLK monotonicity
        is_sorted = df['SCLK'].is_monotonic_increasing
        if is_sorted:
            print(f"  [OK] SCLK is sorted (monotonic)")
        else:
            print(f"  [WARNING] SCLK is not sorted!")
        
        # Check for duplicates
        dup_sclk = df['SCLK'].duplicated().sum()
        if dup_sclk == 0:
            print(f"  [OK] No duplicate SCLK values")
        else:
            print(f"  [WARNING] {dup_sclk} duplicate SCLK values found")
        
        # Check pressure range
        pressure_min = df['PRESSURE'].min()
        pressure_max = df['PRESSURE'].max()
        pressure_mean = df['PRESSURE'].mean()
        pressure_std = df['PRESSURE'].std()
        
        print(f"  Pressure: mean={pressure_mean:.2f} Pa, std={pressure_std:.2f} Pa")
        print(f"  Pressure range: {pressure_min:.2f} to {pressure_max:.2f} Pa")
        
        # Check for outliers (beyond 3 std)
        outliers = ((df['PRESSURE'] < pressure_mean - 3*pressure_std) | 
                   (df['PRESSURE'] > pressure_mean + 3*pressure_std)).sum()
        if outliers > 0:
            print(f"  [WARNING] {outliers} pressure outliers (>3 std)")
        else:
            print(f"  [OK] No extreme pressure outliers")

def main():
    """Main validation function."""
    print("=" * 70)
    print("DATA PREPARATION VALIDATION")
    print("=" * 70)
    
    all_checks_passed = True
    
    # Validate temporal splits
    if not validate_temporal_splits():
        all_checks_passed = False
    
    # Validate windows
    if not validate_windows():
        all_checks_passed = False
    
    # Validate data quality
    validate_data_quality()
    
    print("\n" + "=" * 70)
    if all_checks_passed:
        print("VALIDATION COMPLETE - All checks passed!")
    else:
        print("VALIDATION COMPLETE - Some checks failed. Review warnings above.")
    print("=" * 70)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    exit(main())

