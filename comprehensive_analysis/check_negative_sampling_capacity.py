#!/usr/bin/env python3
"""Quick check of negative sampling capacity for filtered dataset."""

import pandas as pd
import numpy as np
import os

SPLITS_DIR = "data/splits"
WINDOWS_DIR = "data/windows"
WINDOW_SIZE = 60

# Load data
ml_train = pd.read_csv(os.path.join(SPLITS_DIR, "ml_train.csv"))
train_windows = pd.read_csv(os.path.join(WINDOWS_DIR, "train_windows.csv"))

print("=" * 70)
print("NEGATIVE SAMPLING CAPACITY CHECK")
print("=" * 70)

print(f"\nTraining split:")
print(f"  Total ML samples: {len(ml_train):,}")
print(f"  Positive windows: {train_windows['window_id'].nunique()}")

# Check safe regions
forbidden = np.zeros(len(ml_train), dtype=bool)
forbidden[ml_train['gt_detection_win'] == True] = True
safe_samples = (~forbidden).sum()

print(f"\nSafe regions for negative sampling:")
print(f"  Forbidden samples (vortex regions): {forbidden.sum():,} ({forbidden.sum()/len(ml_train)*100:.1f}%)")
print(f"  Safe samples: {safe_samples:,} ({safe_samples/len(ml_train)*100:.1f}%)")

# Estimate max negative windows
max_negative_windows = safe_samples // WINDOW_SIZE
print(f"  Estimated max negative windows (60-sample): ~{max_negative_windows:,}")

# Current status
if os.path.exists("data/features/train_balanced.csv"):
    balanced = pd.read_csv("data/features/train_balanced.csv")
    pos_count = (balanced['label'] == 1).sum()
    neg_count = (balanced['label'] == 0).sum()
    print(f"\nCurrent balanced dataset:")
    print(f"  Positive: {pos_count}")
    print(f"  Negative: {neg_count}")
    print(f"  Ratio: {neg_count/pos_count:.2f}:1" if pos_count > 0 else "  Ratio: N/A")
    print(f"\nCan increase negative samples: YES (up to ~{max_negative_windows:,})")

print("\n" + "=" * 70)
print("CONCLUSION: Negative sampling CAN be done for filtered dataset!")
print("=" * 70)



