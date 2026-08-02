#!/usr/bin/env python3
"""
Verify Global Statistics Consistency
====================================

This script checks if global statistics (mean, std) are computed consistently
across train/val/test splits. This is critical for feature engineering consistency.

Issues to check:
1. Are global statistics computed from windows or full splits?
2. Are they consistent across train/val/test?
3. Does this cause feature distribution shifts?
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_WINDOWS = "datasets/train_balanced.csv"
VAL_WINDOWS = "datasets/val_balanced.csv"
TEST_WINDOWS = "datasets/test_balanced.csv"

TRAIN_SPLIT = "datasets/temporal_splits/ml_train.csv"
VAL_SPLIT = "datasets/temporal_splits/ml_val.csv"
TEST_SPLIT = "datasets/temporal_splits/ml_test.csv"

TRAIN_FEATURES = "datasets/train_features.csv"
VAL_FEATURES = "datasets/val_features.csv"
TEST_FEATURES = "datasets/test_features.csv"

# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def check_global_statistics_from_windows():
    """Check how global statistics are computed from windows."""
    print("=" * 70)
    print("CHECKING GLOBAL STATISTICS FROM WINDOWS")
    print("=" * 70)
    
    results = {}
    
    for split_name, windows_file in [("train", TRAIN_WINDOWS), 
                                     ("val", VAL_WINDOWS), 
                                     ("test", TEST_WINDOWS)]:
        if not os.path.exists(windows_file):
            print(f"[WARNING] {windows_file} not found, skipping...")
            continue
        
        windows_df = pd.read_csv(windows_file)
        global_mean = windows_df['PRESSURE'].mean()
        global_std = windows_df['PRESSURE'].std()
        
        results[split_name] = {
            'mean': global_mean,
            'std': global_std,
            'samples': len(windows_df)
        }
        
        print(f"\n{split_name.upper()} (from windows):")
        print(f"  Samples: {len(windows_df):,}")
        print(f"  Global mean: {global_mean:.4f} Pa")
        print(f"  Global std:  {global_std:.4f} Pa")
    
    # Check consistency
    if len(results) >= 2:
        print("\n" + "-" * 70)
        print("CONSISTENCY CHECK:")
        print("-" * 70)
        
        means = [r['mean'] for r in results.values()]
        stds = [r['std'] for r in results.values()]
        
        mean_range = max(means) - min(means)
        std_range = max(stds) - min(stds)
        
        print(f"  Mean range: {mean_range:.4f} Pa (max - min)")
        print(f"  Std range:  {std_range:.4f} Pa (max - min)")
        
        if mean_range > 1.0 or std_range > 1.0:
            print(f"  [WARNING] Large variation detected! This could cause inconsistency.")
        else:
            print(f"  [OK] Statistics are relatively consistent.")
    
    return results

def check_global_statistics_from_splits():
    """Check global statistics from full temporal splits."""
    print("\n" + "=" * 70)
    print("CHECKING GLOBAL STATISTICS FROM FULL SPLITS")
    print("=" * 70)
    
    results = {}
    
    for split_name, split_file in [("train", TRAIN_SPLIT), 
                                   ("val", VAL_SPLIT), 
                                   ("test", TEST_SPLIT)]:
        if not os.path.exists(split_file):
            print(f"[WARNING] {split_file} not found, skipping...")
            continue
        
        split_df = pd.read_csv(split_file)
        global_mean = split_df['PRESSURE'].mean()
        global_std = split_df['PRESSURE'].std()
        
        results[split_name] = {
            'mean': global_mean,
            'std': global_std,
            'samples': len(split_df)
        }
        
        print(f"\n{split_name.upper()} (from full split):")
        print(f"  Samples: {len(split_df):,}")
        print(f"  Global mean: {global_mean:.4f} Pa")
        print(f"  Global std:  {global_std:.4f} Pa")
    
    # Check consistency
    if len(results) >= 2:
        print("\n" + "-" * 70)
        print("CONSISTENCY CHECK:")
        print("-" * 70)
        
        means = [r['mean'] for r in results.values()]
        stds = [r['std'] for r in results.values()]
        
        mean_range = max(means) - min(means)
        std_range = max(stds) - min(stds)
        
        print(f"  Mean range: {mean_range:.4f} Pa (max - min)")
        print(f"  Std range:  {std_range:.4f} Pa (max - min)")
        
        if mean_range > 1.0 or std_range > 1.0:
            print(f"  [WARNING] Large variation detected!")
        else:
            print(f"  [OK] Statistics are relatively consistent.")
    
    return results

def check_feature_distributions():
    """Check if feature distributions are consistent across splits."""
    print("\n" + "=" * 70)
    print("CHECKING FEATURE DISTRIBUTIONS")
    print("=" * 70)
    
    results = {}
    
    for split_name, features_file in [("train", TRAIN_FEATURES), 
                                      ("val", VAL_FEATURES), 
                                      ("test", TEST_FEATURES)]:
        if not os.path.exists(features_file):
            print(f"[WARNING] {features_file} not found, skipping...")
            continue
        
        features_df = pd.read_csv(features_file)
        
        # Check key features that use global statistics
        key_features = ['mean', 'std', 'pressure_drop', 'min_zscore']
        
        results[split_name] = {}
        for feature in key_features:
            if feature in features_df.columns:
                results[split_name][feature] = {
                    'mean': features_df[feature].mean(),
                    'std': features_df[feature].std(),
                    'min': features_df[feature].min(),
                    'max': features_df[feature].max()
                }
    
    # Compare distributions
    if len(results) >= 2:
        print("\n" + "-" * 70)
        print("FEATURE DISTRIBUTION COMPARISON:")
        print("-" * 70)
        
        for feature in ['mean', 'std', 'pressure_drop', 'min_zscore']:
            if all(feature in r for r in results.values()):
                print(f"\n{feature}:")
                for split_name, stats in results.items():
                    if feature in stats:
                        print(f"  {split_name:6s}: mean={stats[feature]['mean']:8.4f}, "
                              f"std={stats[feature]['std']:8.4f}, "
                              f"range=[{stats[feature]['min']:6.2f}, {stats[feature]['max']:6.2f}]")
                
                # Check for large differences
                means = [r[feature]['mean'] for r in results.values() if feature in r]
                stds = [r[feature]['std'] for r in results.values() if feature in r]
                
                if means and stds:
                    mean_range = max(means) - min(means)
                    std_range = max(stds) - min(stds)
                    
                    if mean_range > 0.1 * abs(np.mean(means)) or std_range > 0.1 * abs(np.mean(stds)):
                        print(f"    [WARNING] Large variation: mean_range={mean_range:.4f}, std_range={std_range:.4f}")
                    else:
                        print(f"    [OK] Distributions are relatively consistent")
    
    return results

def recommend_fix(window_stats, split_stats, feature_stats):
    """Recommend whether Fix 1 is needed."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    # Check if there's significant variation
    if window_stats and len(window_stats) >= 2:
        window_means = [r['mean'] for r in window_stats.values()]
        window_stds = [r['std'] for r in window_stats.values()]
        window_mean_range = max(window_means) - min(window_means)
        window_std_range = max(window_stds) - min(window_stds)
    else:
        window_mean_range = 0
        window_std_range = 0
    
    if split_stats and len(split_stats) >= 2:
        split_means = [r['mean'] for r in split_stats.values()]
        split_stds = [r['std'] for r in split_stats.values()]
        split_mean_range = max(split_means) - min(split_means)
        split_std_range = max(split_stds) - min(split_stds)
    else:
        split_mean_range = 0
        split_std_range = 0
    
    # Check feature consistency (especially min_zscore which uses global stats)
    min_zscore_inconsistent = False
    if feature_stats and len(feature_stats) >= 2:
        if all('min_zscore' in r for r in feature_stats.values()):
            zscore_means = [r['min_zscore']['mean'] for r in feature_stats.values()]
            zscore_range = max(zscore_means) - min(zscore_means)
            if zscore_range > 0.5:  # Significant variation
                min_zscore_inconsistent = True
    
    print(f"\nAnalysis:")
    print(f"  Window stats variation: mean_range={window_mean_range:.4f}, std_range={window_std_range:.4f}")
    print(f"  Split stats variation:  mean_range={split_mean_range:.4f}, std_range={split_std_range:.4f}")
    print(f"  min_zscore inconsistency: {min_zscore_inconsistent}")
    
    # Decision logic
    needs_fix = False
    reasons = []
    
    if window_mean_range > 1.0 or window_std_range > 1.0:
        needs_fix = True
        reasons.append("Large variation in window-based global statistics")
    
    if min_zscore_inconsistent:
        needs_fix = True
        reasons.append("min_zscore feature shows inconsistent distributions")
    
    if split_mean_range < window_mean_range:
        needs_fix = True
        reasons.append("Split-based stats are more consistent than window-based")
    
    print(f"\n{'='*70}")
    if needs_fix:
        print("RECOMMENDATION: APPLY FIX 1")
        print("="*70)
        print("\nReasons:")
        for i, reason in enumerate(reasons, 1):
            print(f"  {i}. {reason}")
        print("\nAction: Update feature_engineering.py to use training split statistics")
        print("        for all splits (train, val, test) to ensure consistency.")
    else:
        print("RECOMMENDATION: NO FIX NEEDED")
        print("="*70)
        print("\nGlobal statistics are relatively consistent across splits.")
        print("However, using training set statistics for all splits is still")
        print("a best practice to ensure perfect consistency.")
    
    return needs_fix, reasons

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main verification pipeline."""
    print("=" * 70)
    print("GLOBAL STATISTICS CONSISTENCY VERIFICATION")
    print("=" * 70)
    print("\nThis script checks if global statistics are computed consistently")
    print("across train/val/test splits, which is critical for feature engineering.")
    print("=" * 70)
    
    # Step 1: Check statistics from windows (current approach)
    window_stats = check_global_statistics_from_windows()
    
    # Step 2: Check statistics from full splits (better approach)
    split_stats = check_global_statistics_from_splits()
    
    # Step 3: Check feature distributions
    feature_stats = check_feature_distributions()
    
    # Step 4: Recommend fix
    needs_fix, reasons = recommend_fix(window_stats, split_stats, feature_stats)
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    return 0 if not needs_fix else 1

if __name__ == "__main__":
    exit(main())
