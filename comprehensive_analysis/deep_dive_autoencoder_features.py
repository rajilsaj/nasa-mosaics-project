#!/usr/bin/env python3
"""
Deep Dive: Autoencoder Features Investigation
============================================

Check why autoencoder features have NaN for positive samples.
"""

import pandas as pd
import numpy as np
import os

FEATURES_DIR = "data/features"

# Load training data
train_file = os.path.join(FEATURES_DIR, "train_balanced.csv")
train_df = pd.read_csv(train_file)

print("=" * 70)
print("AUTOENCODER FEATURES DEEP DIVE")
print("=" * 70)

# Check autoencoder features
ae_features = [col for col in train_df.columns if 'autoencoder' in col.lower() or 'ae_' in col.lower()]

print(f"\nFound {len(ae_features)} autoencoder-related features:")
for feat in ae_features:
    print(f"  - {feat}")

# Check for NaN values
print("\n" + "=" * 70)
print("NaN VALUE ANALYSIS")
print("=" * 70)

for feat in ae_features:
    if feat in train_df.columns:
        nan_count = train_df[feat].isna().sum()
        pos_nan = train_df[train_df['label'] == 1][feat].isna().sum()
        neg_nan = train_df[train_df['label'] == 0][feat].isna().sum()
        
        print(f"\n{feat}:")
        print(f"  Total NaN: {nan_count} ({nan_count/len(train_df)*100:.1f}%)")
        print(f"  Positive NaN: {pos_nan} ({pos_nan/(train_df['label']==1).sum()*100:.1f}%)")
        print(f"  Negative NaN: {neg_nan} ({neg_nan/(train_df['label']==0).sum()*100:.1f}%)")
        
        # Show value distribution
        pos_vals = train_df[train_df['label'] == 1][feat].dropna()
        neg_vals = train_df[train_df['label'] == 0][feat].dropna()
        
        if len(pos_vals) > 0:
            print(f"  Positive (non-NaN): min={pos_vals.min():.4f}, max={pos_vals.max():.4f}, mean={pos_vals.mean():.4f}")
        if len(neg_vals) > 0:
            print(f"  Negative (non-NaN): min={neg_vals.min():.4f}, max={neg_vals.max():.4f}, mean={neg_vals.mean():.4f}")

# Check if NaN is causing perfect separation
print("\n" + "=" * 70)
print("PERFECT SEPARATION CHECK")
print("=" * 70)

for feat in ae_features:
    if feat in train_df.columns:
        # Check if all positive have NaN and all negative have values (or vice versa)
        pos_has_nan = train_df[train_df['label'] == 1][feat].isna().all()
        neg_has_nan = train_df[train_df['label'] == 0][feat].isna().all()
        
        if pos_has_nan and not neg_has_nan:
            print(f"\n[CRITICAL] {feat}:")
            print(f"  ALL positive samples have NaN")
            print(f"  ALL negative samples have values")
            print(f"  This creates PERFECT SEPARATION (data leakage!)")
        elif neg_has_nan and not pos_has_nan:
            print(f"\n[CRITICAL] {feat}:")
            print(f"  ALL negative samples have NaN")
            print(f"  ALL positive samples have values")
            print(f"  This creates PERFECT SEPARATION (data leakage!)")

# Check ae_gt_agreement feature specifically
if 'ae_gt_agreement' in train_df.columns:
    print("\n" + "=" * 70)
    print("ae_gt_agreement FEATURE ANALYSIS")
    print("=" * 70)
    
    print(f"\nValue distribution:")
    print(train_df['ae_gt_agreement'].value_counts())
    
    print(f"\nBy label:")
    print(train_df.groupby('label')['ae_gt_agreement'].value_counts())




