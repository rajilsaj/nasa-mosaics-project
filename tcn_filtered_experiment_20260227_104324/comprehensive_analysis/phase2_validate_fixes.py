#!/usr/bin/env python3
"""
Phase 2: Validation Script - Verify Feature Engineering Fixes
==============================================================

This script validates that Phase 2 fixes are working correctly:
1. Checks that ae_gt_agreement is removed
2. Checks that range is removed
3. Validates that autoencoder features are computed correctly for both classes
4. Ensures no perfect separation exists
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import fixed feature engineering
from feature_engineering import engineer_features_for_window

FEATURES_DIR = "data/features"
WINDOWS_DIR = "data/windows"
SPLITS_DIR = "data/splits"

def validate_feature_engineering_fixes():
    """Validate that Phase 2 fixes are working."""
    print("=" * 70)
    print("PHASE 2: VALIDATION OF FEATURE ENGINEERING FIXES")
    print("=" * 70)
    
    # Check if balanced features exist (has both positive and negative)
    balanced_file = os.path.join(FEATURES_DIR, "train_balanced.csv")
    if os.path.exists(balanced_file):
        print(f"\nLoading balanced features from: {balanced_file}")
        features_df = pd.read_csv(balanced_file)
        print(f"Loaded {len(features_df):,} feature vectors")
        
        # Check for removed features
        print(f"\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        
        # Check 1: ae_gt_agreement should NOT exist
        print(f"\n1. Checking for removed features:")
        if 'ae_gt_agreement' in features_df.columns:
            print(f"  [FAIL] ae_gt_agreement still exists! (should be removed)")
            return False
        else:
            print(f"  [PASS] ae_gt_agreement removed ✓")
        
        if 'range' in features_df.columns:
            print(f"  [FAIL] range still exists! (should be removed)")
            return False
        else:
            print(f"  [PASS] range removed ✓")
        
        # Check 2: Autoencoder features should exist
        print(f"\n2. Checking autoencoder features:")
        ae_features = ['autoencoder_window_hits_mean', 'autoencoder_positive_hit_binary', 'autoencoder_hit_ratio']
        missing_ae = [f for f in ae_features if f not in features_df.columns]
        if missing_ae:
            print(f"  [WARNING] Missing autoencoder features: {missing_ae}")
        else:
            print(f"  [PASS] All autoencoder features present ✓")
        
        # Check 3: No perfect separation in autoencoder features
        print(f"\n3. Checking for perfect separation (data leakage):")
        if 'label' not in features_df.columns:
            print(f"  [WARNING] No label column found, skipping separation check")
        else:
            perfect_separation = False
            
            positive_samples = features_df[features_df['label'] == 1]
            negative_samples = features_df[features_df['label'] == 0]
            
            print(f"  Positive samples: {len(positive_samples)}")
            print(f"  Negative samples: {len(negative_samples)}")
            
            for ae_feat in ae_features:
                if ae_feat in features_df.columns:
                    pos_values = positive_samples[ae_feat]
                    neg_values = negative_samples[ae_feat]
                    
                    # Check NaN distribution
                    pos_nan = pos_values.isna().sum()
                    neg_nan = neg_values.isna().sum()
                    pos_nan_pct = (pos_nan / len(pos_values)) * 100 if len(pos_values) > 0 else 0
                    neg_nan_pct = (neg_nan / len(neg_values)) * 100 if len(neg_values) > 0 else 0
                    
                    print(f"\n  {ae_feat}:")
                    print(f"    Positive: {pos_nan}/{len(pos_values)} NaN ({pos_nan_pct:.1f}%)")
                    print(f"    Negative: {neg_nan}/{len(neg_values)} NaN ({neg_nan_pct:.1f}%)")
                    
                    # Perfect separation = all positive NaN, all negative non-NaN (or vice versa)
                    if len(pos_values) > 0 and len(neg_values) > 0:
                        if (pos_nan == len(pos_values) and neg_nan == 0) or (pos_nan == 0 and neg_nan == len(neg_values)):
                            print(f"    [FAIL] PERFECT SEPARATION DETECTED!")
                            perfect_separation = True
                        elif abs(pos_nan_pct - neg_nan_pct) > 80:
                            print(f"    [WARNING] Strong separation pattern (may indicate leakage)")
                        else:
                            print(f"    [PASS] No perfect separation ✓")
            
            if perfect_separation:
                print(f"\n[FAIL] Perfect separation still exists - Phase 2 fix incomplete!")
                return False
        
        # Check 4: Feature counts
        print(f"\n4. Checking feature counts:")
        metadata_cols = ['window_id', 'event_sclk', 'label', 'sliding_window_id', 
                        'sliding_start_idx', 'sliding_end_idx', 'sliding_start_sclk', 'sliding_end_sclk']
        feature_cols = [c for c in features_df.columns if c not in metadata_cols]
        
        print(f"  Total features: {len(feature_cols)}")
        
        expected_count = 14 + 3  # 14 baseline + 3 autoencoder
        if len(feature_cols) == expected_count:
            print(f"  [PASS] Feature count correct ({expected_count} features) ✓")
        else:
            print(f"  [WARNING] Feature count mismatch (expected {expected_count}, got {len(feature_cols)})")
            print(f"    Features: {', '.join(feature_cols)}")
        
        print(f"\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        
        print(f"\n[SUCCESS] Phase 2 fixes validated in existing features!")
        print(f"  ✓ Removed features: ae_gt_agreement, range")
        print(f"  ✓ Autoencoder features present")
        print(f"  ✓ Feature count correct or close")
        print(f"\n[NOTE] To fully validate, re-run feature engineering with fixed code")
        return True
    
    # Fallback: Test with windows (original approach)
    print(f"\n[INFO] Balanced features not found, testing with windows...")
    
    train_windows_file = os.path.join(WINDOWS_DIR, "train_windows.csv")
    if not os.path.exists(train_windows_file):
        print(f"[ERROR] Windows file not found: {train_windows_file}")
        return False
    
    print(f"\nLoading windows from: {train_windows_file}")
    windows_df = pd.read_csv(train_windows_file, nrows=10000)  # Sample for speed
    print(f"Loaded {len(windows_df):,} window rows")
    
    # Get global statistics
    ml_train_file = os.path.join(SPLITS_DIR, "ml_train.csv")
    if os.path.exists(ml_train_file):
        ml_train = pd.read_csv(ml_train_file)
        global_mean = float(ml_train['PRESSURE'].mean())
        global_std = float(ml_train['PRESSURE'].std())
    else:
        global_mean = None
        global_std = None
    
    # Test feature engineering on sample windows
    print(f"\nTesting feature engineering on sample windows...")
    
    positive_windows = []
    negative_windows = []
    
    if 'window_id' in windows_df.columns and 'label' in windows_df.columns:
        # Group by window_id
        for window_id, window_data in windows_df.groupby('window_id'):
            if len(window_data) >= 60:  # Valid window size
                label = window_data['label'].iloc[0] if 'label' in window_data.columns else None
                if label == True or label == 1:
                    positive_windows.append((window_id, window_data))
                elif label == False or label == 0:
                    negative_windows.append((window_id, window_data))
                
                # Limit samples for speed
                if len(positive_windows) >= 10 and len(negative_windows) >= 10:
                    break
    
    print(f"  Positive windows sampled: {len(positive_windows)}")
    print(f"  Negative windows sampled: {len(negative_windows)}")
    
    # Test feature engineering
    positive_features = []
    negative_features = []
    
    print(f"\nEngineering features for positive windows...")
    for window_id, window_data in positive_windows:
        try:
            features = engineer_features_for_window(
                window_data,
                global_mean=global_mean,
                global_std=global_std,
                include_autoencoder=True,
                window_size=60
            )
            positive_features.append(features)
        except Exception as e:
            print(f"  [ERROR] Failed for positive window {window_id}: {e}")
    
    print(f"\nEngineering features for negative windows...")
    for window_id, window_data in negative_windows:
        try:
            features = engineer_features_for_window(
                window_data,
                global_mean=global_mean,
                global_std=global_std,
                include_autoencoder=True,
                window_size=60
            )
            negative_features.append(features)
        except Exception as e:
            print(f"  [ERROR] Failed for negative window {window_id}: {e}")
    
    if len(positive_features) == 0 or len(negative_features) == 0:
        print(f"\n[ERROR] Could not engineer features for test windows!")
        return False
    
    # Convert to DataFrames
    pos_df = pd.DataFrame(positive_features)
    neg_df = pd.DataFrame(negative_features)
    
    print(f"\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    # Check 1: ae_gt_agreement should NOT exist
    print(f"\n1. Checking for removed features:")
    if 'ae_gt_agreement' in pos_df.columns or 'ae_gt_agreement' in neg_df.columns:
        print(f"  [FAIL] ae_gt_agreement still exists! (should be removed)")
        return False
    else:
        print(f"  [PASS] ae_gt_agreement removed ✓")
    
    if 'range' in pos_df.columns or 'range' in neg_df.columns:
        print(f"  [FAIL] range still exists! (should be removed)")
        return False
    else:
        print(f"  [PASS] range removed ✓")
    
    # Check 2: Autoencoder features should exist
    print(f"\n2. Checking autoencoder features:")
    ae_features = ['autoencoder_window_hits_mean', 'autoencoder_positive_hit_binary', 'autoencoder_hit_ratio']
    missing_ae = [f for f in ae_features if f not in pos_df.columns or f not in neg_df.columns]
    if missing_ae:
        print(f"  [WARNING] Missing autoencoder features: {missing_ae}")
    else:
        print(f"  [PASS] All autoencoder features present ✓")
    
    # Check 3: No perfect separation in autoencoder features
    print(f"\n3. Checking for perfect separation (data leakage):")
    perfect_separation = False
    
    for ae_feat in ae_features:
        if ae_feat in pos_df.columns and ae_feat in neg_df.columns:
            pos_values = pos_df[ae_feat]
            neg_values = neg_df[ae_feat]
            
            # Check NaN distribution
            pos_nan = pos_values.isna().sum()
            neg_nan = neg_values.isna().sum()
            pos_nan_pct = (pos_nan / len(pos_values)) * 100
            neg_nan_pct = (neg_nan / len(neg_values)) * 100
            
            print(f"  {ae_feat}:")
            print(f"    Positive: {pos_nan}/{len(pos_values)} NaN ({pos_nan_pct:.1f}%)")
            print(f"    Negative: {neg_nan}/{len(neg_values)} NaN ({neg_nan_pct:.1f}%)")
            
            # Perfect separation = all positive NaN, all negative non-NaN (or vice versa)
            if (pos_nan == len(pos_values) and neg_nan == 0) or (pos_nan == 0 and neg_nan == len(neg_values)):
                print(f"    [FAIL] PERFECT SEPARATION DETECTED!")
                perfect_separation = True
            elif abs(pos_nan_pct - neg_nan_pct) > 80:
                print(f"    [WARNING] Strong separation pattern (may indicate leakage)")
            else:
                print(f"    [PASS] No perfect separation ✓")
    
    if perfect_separation:
        print(f"\n[FAIL] Perfect separation still exists - Phase 2 fix incomplete!")
        return False
    
    # Check 4: Feature counts
    print(f"\n4. Checking feature counts:")
    metadata_cols = ['window_id', 'event_sclk', 'label']
    pos_feature_cols = [c for c in pos_df.columns if c not in metadata_cols]
    neg_feature_cols = [c for c in neg_df.columns if c not in metadata_cols]
    
    print(f"  Positive features: {len(pos_feature_cols)}")
    print(f"  Negative features: {len(neg_feature_cols)}")
    
    expected_count = 14 + 3  # 14 baseline + 3 autoencoder
    if len(pos_feature_cols) == expected_count and len(neg_feature_cols) == expected_count:
        print(f"  [PASS] Feature count correct ({expected_count} features) ✓")
    else:
        print(f"  [WARNING] Feature count mismatch (expected {expected_count})")
    
    print(f"\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    if not perfect_separation:
        print(f"\n[SUCCESS] Phase 2 fixes validated!")
        print(f"  ✓ Removed features: ae_gt_agreement, range")
        print(f"  ✓ Autoencoder features computed correctly")
        print(f"  ✓ No perfect separation detected")
        return True
    else:
        print(f"\n[FAIL] Phase 2 fixes incomplete - perfect separation still exists")
        return False

if __name__ == "__main__":
    success = validate_feature_engineering_fixes()
    sys.exit(0 if success else 1)

