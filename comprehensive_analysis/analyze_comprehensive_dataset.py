#!/usr/bin/env python3
"""
Initial Analysis of comprehensive_filtered_data_optimized.csv
Working in comprehensive_analysis/ folder
"""

import pandas as pd
import numpy as np
import os

# Path to comprehensive dataset (one level up)
COMPREHENSIVE_DATA = "../comprehensive_filtered_data_optimized.csv"

print("=" * 80)
print("COMPREHENSIVE DATASET ANALYSIS")
print("=" * 80)

# Load data
print(f"\nLoading {COMPREHENSIVE_DATA}...")
df = pd.read_csv(COMPREHENSIVE_DATA)

print(f"[OK] Loaded {len(df):,} samples")
print(f"[OK] Columns: {list(df.columns)}")

# Basic stats
print(f"\n{'='*80}")
print("DATASET STATISTICS")
print(f"{'='*80}")
print(f"Total samples: {len(df):,}")
print(f"Time span: Sol {df['sol'].min()} to {df['sol'].max()} ({df['sol'].max() - df['sol'].min()} sols)")
print(f"SCLK range: {df['SCLK'].min()} to {df['SCLK'].max()}")

# Class distribution
vortex_count = df['gt_detection_win'].sum()
non_vortex_count = (~df['gt_detection_win']).sum()
print(f"\nClass Distribution:")
print(f"  Vortex events: {vortex_count:,} ({vortex_count/len(df)*100:.3f}%)")
print(f"  Non-vortex: {non_vortex_count:,} ({non_vortex_count/len(df)*100:.3f}%)")
print(f"  Imbalance ratio: {non_vortex_count/vortex_count:.1f}:1 (Neg:Pos)")

# Autoencoder features
if 'autoencoder_positive_hit' in df.columns:
    ae_pos = df['autoencoder_positive_hit'].sum()
    print(f"\nAutoencoder Features:")
    print(f"  autoencoder_positive_hit: {ae_pos:,} ({ae_pos/len(df)*100:.2f}% positive)")
    print(f"  autoencoder_window_hits: mean={df['autoencoder_window_hits'].mean():.2f}, max={df['autoencoder_window_hits'].max()}")

# Pressure stats
print(f"\nPressure Statistics:")
print(f"  Mean: {df['PRESSURE'].mean():.2f} Pa")
print(f"  Std: {df['PRESSURE'].std():.2f} Pa")
print(f"  Range: {df['PRESSURE'].min():.2f} - {df['PRESSURE'].max():.2f} Pa")

print(f"\n{'='*80}")
print("Dataset ready for analysis!")
print(f"{'='*80}")

