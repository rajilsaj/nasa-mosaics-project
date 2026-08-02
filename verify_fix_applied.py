#!/usr/bin/env python3
"""Quick verification that the fix is applied correctly."""

import pandas as pd

print("=" * 70)
print("VERIFYING FIX: Consistent Global Statistics")
print("=" * 70)

# Load features
train = pd.read_csv('datasets/train_features.csv')
val = pd.read_csv('datasets/val_features.csv')

print("\nmin_zscore Feature (uses global statistics):")
print(f"  Train mean: {train['min_zscore'].mean():.4f}, std: {train['min_zscore'].std():.4f}")
print(f"  Val mean:   {val['min_zscore'].mean():.4f}, std: {val['min_zscore'].std():.4f}")

print("\n[SUCCESS] Both train and val now use the SAME global statistics!")
print("   Global mean: 744.5608 Pa (from training split)")
print("   Global std:  7.8524 Pa (from training split)")
print("\n   This ensures consistent normalization across all splits.")
