#!/usr/bin/env python3
"""Verify feature importance matches the model trained on ML dataset."""

import pandas as pd
import os

print("=" * 70)
print("FEATURE IMPORTANCE VERIFICATION")
print("=" * 70)

# Expected 15 features from ML dataset model
EXPECTED_FEATURES = [
    'overall_slope', 'first_half_slope', 'second_half_slope', 'trend_consistency',
    'pressure_drop', 'drop_rate', 'min_position',
    'mean', 'std', 'range',
    'first_half_mean', 'second_half_mean', 'mean_ratio',
    'min_zscore', 'anomaly_strength'
]

# Load feature importance
importance_file = "results/feature_importance.csv"
if os.path.exists(importance_file):
    importance_df = pd.read_csv(importance_file)
    actual_features = importance_df['feature'].tolist()
    
    print(f"\n1. Feature Importance CSV contains {len(actual_features)} features:")
    for i, feat in enumerate(actual_features, 1):
        print(f"   {i:2d}. {feat}")
    
    print(f"\n2. Expected {len(EXPECTED_FEATURES)} features from ML dataset model:")
    for i, feat in enumerate(EXPECTED_FEATURES, 1):
        print(f"   {i:2d}. {feat}")
    
    # Check match
    match = set(actual_features) == set(EXPECTED_FEATURES)
    print(f"\n3. Feature Match Check:")
    print(f"   ✓ Features match: {match}")
    
    if match:
        print(f"\n   ✅ VERIFIED: Feature importance reflects the 15 features")
        print(f"      used in the Random Forest model trained on ML dataset")
    else:
        missing = set(EXPECTED_FEATURES) - set(actual_features)
        extra = set(actual_features) - set(EXPECTED_FEATURES)
        if missing:
            print(f"   ⚠ Missing features: {missing}")
        if extra:
            print(f"   ⚠ Extra features: {extra}")
    
    # Check if train_features.csv exists and verify
    if os.path.exists("datasets/train_features.csv"):
        train_df = pd.read_csv("datasets/train_features.csv", nrows=1)
        exclude_cols = ['window_id', 'event_sclk', 'label']
        train_features = [c for c in train_df.columns if c not in exclude_cols]
        
        print(f"\n4. Training Features CSV contains {len(train_features)} features:")
        for i, feat in enumerate(train_features, 1):
            print(f"   {i:2d}. {feat}")
        
        train_match = set(train_features) == set(EXPECTED_FEATURES)
        print(f"\n   ✓ Training features match expected: {train_match}")
        
        importance_match = set(actual_features) == set(train_features)
        print(f"   ✓ Importance features match training features: {importance_match}")
        
        if match and train_match and importance_match:
            print(f"\n   ✅ ALL VERIFIED: Feature importance correctly reflects")
            print(f"      the model trained on ML dataset with 15 features")
    
else:
    print(f"ERROR: {importance_file} not found")

print("\n" + "=" * 70)
