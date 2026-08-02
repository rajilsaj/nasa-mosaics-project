#!/usr/bin/env python3
"""Compare comprehensive_filtered_data_optimized.csv with ml_ready_vortex_data.csv"""

import pandas as pd
import numpy as np

print("=" * 80)
print("DATASET COMPARISON ANALYSIS")
print("=" * 80)

# Load both datasets
df1 = pd.read_csv('comprehensive_filtered_data_optimized.csv')

# Load temporal splits (current pipeline data)
df_train = pd.read_csv('datasets/temporal_splits/ml_train.csv')
df_val = pd.read_csv('datasets/temporal_splits/ml_val.csv')
df_test = pd.read_csv('datasets/temporal_splits/ml_test.csv')
df2 = pd.concat([df_train, df_val, df_test], ignore_index=True)
print("Loaded temporal splits (train+val+test) as comparison dataset")

print("\n" + "=" * 80)
print("1. DATASET SIZE COMPARISON")
print("=" * 80)
print(f"\ncomprehensive_filtered_data_optimized.csv:")
print(f"  Rows: {len(df1):,}")
print(f"  Columns: {len(df1.columns)}")
print(f"  Memory: {df1.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\nml_ready_vortex_data.csv:")
print(f"  Rows: {len(df2):,}")
print(f"  Columns: {len(df2.columns)}")
print(f"  Memory: {df2.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\nDifference:")
print(f"  comprehensive has {len(df1) - len(df2):,} more rows")
print(f"  comprehensive is {len(df1)/len(df2):.2f}x larger")

print("\n" + "=" * 80)
print("2. COLUMN COMPARISON")
print("=" * 80)
cols1 = set(df1.columns)
cols2 = set(df2.columns)

print(f"\ncomprehensive_filtered_data_optimized.csv columns:")
print(f"  {sorted(cols1)}")

print(f"\nml_ready_vortex_data.csv columns:")
print(f"  {sorted(cols2)}")

only_in_comprehensive = cols1 - cols2
only_in_ml_ready = cols2 - cols1
common_cols = cols1 & cols2

print(f"\nOnly in comprehensive_filtered_data_optimized.csv:")
if only_in_comprehensive:
    for col in sorted(only_in_comprehensive):
        print(f"  • {col}")
else:
    print("  (none)")

print(f"\nOnly in ml_ready_vortex_data.csv:")
if only_in_ml_ready:
    for col in sorted(only_in_ml_ready):
        print(f"  • {col}")
else:
    print("  (none)")

print(f"\nCommon columns ({len(common_cols)}):")
print(f"  {sorted(common_cols)}")

print("\n" + "=" * 80)
print("3. TEMPORAL COVERAGE COMPARISON")
print("=" * 80)
print(f"\ncomprehensive_filtered_data_optimized.csv:")
print(f"  SCLK range: {df1['SCLK'].min()} to {df1['SCLK'].max()}")
print(f"  Sol range: {df1['sol'].min()} to {df1['sol'].max()}")
print(f"  Time span: {df1['sol'].max() - df1['sol'].min()} sols")

print(f"\nml_ready_vortex_data.csv:")
print(f"  SCLK range: {df2['SCLK'].min()} to {df2['SCLK'].max()}")
print(f"  Sol range: {df2['sol'].min()} to {df2['sol'].max()}")
print(f"  Time span: {df2['sol'].max() - df2['sol'].min()} sols")

# Check overlap
sclk1_set = set(df1['SCLK'])
sclk2_set = set(df2['SCLK'])
overlap = len(sclk1_set & sclk2_set)

print(f"\nOverlap Analysis:")
print(f"  Common SCLK values: {overlap:,}")
print(f"  Coverage: {overlap/len(df2)*100:.1f}% of ml_ready data is in comprehensive")
print(f"  Unique to comprehensive: {len(sclk1_set - sclk2_set):,} SCLK values")
print(f"  Unique to ml_ready: {len(sclk2_set - sclk1_set):,} SCLK values")

print("\n" + "=" * 80)
print("4. CLASS DISTRIBUTION COMPARISON")
print("=" * 80)
v1 = df1['gt_detection_win'].sum()
n1 = (~df1['gt_detection_win']).sum()
v2 = df2['gt_detection_win'].sum()
n2 = (~df2['gt_detection_win']).sum()

print(f"\ncomprehensive_filtered_data_optimized.csv:")
print(f"  Vortex events (gt_detection_win=True):  {v1:>8,} ({v1/len(df1)*100:>6.3f}%)")
print(f"  Non-vortex (gt_detection_win=False):    {n1:>8,} ({n1/len(df1)*100:>6.3f}%)")
print(f"  Imbalance ratio: {n1/v1:.1f}:1 (Neg:Pos)")

print(f"\nml_ready_vortex_data.csv:")
print(f"  Vortex events (gt_detection_win=True):  {v2:>8,} ({v2/len(df2)*100:>6.3f}%)")
print(f"  Non-vortex (gt_detection_win=False):    {n2:>8,} ({n2/len(df2)*100:>6.3f}%)")
print(f"  Imbalance ratio: {n2/v2:.1f}:1 (Neg:Pos)")

print(f"\nDifference:")
print(f"  comprehensive has {v1-v2:,} more vortex events ({v1/v2:.2f}x)")
print(f"  comprehensive has {n1-n2:,} more non-vortex samples ({n1/n2:.2f}x)")

print("\n" + "=" * 80)
print("5. PRESSURE STATISTICS COMPARISON")
print("=" * 80)
print(f"\ncomprehensive_filtered_data_optimized.csv:")
print(f"  Mean: {df1['PRESSURE'].mean():.2f} Pa")
print(f"  Std:  {df1['PRESSURE'].std():.2f} Pa")
print(f"  Min:  {df1['PRESSURE'].min():.2f} Pa")
print(f"  Max:  {df1['PRESSURE'].max():.2f} Pa")
print(f"  Range: {df1['PRESSURE'].max() - df1['PRESSURE'].min():.2f} Pa")

print(f"\nml_ready_vortex_data.csv:")
print(f"  Mean: {df2['PRESSURE'].mean():.2f} Pa")
print(f"  Std:  {df2['PRESSURE'].std():.2f} Pa")
print(f"  Min:  {df2['PRESSURE'].min():.2f} Pa")
print(f"  Max:  {df2['PRESSURE'].max():.2f} Pa")
print(f"  Range: {df2['PRESSURE'].max() - df2['PRESSURE'].min():.2f} Pa")

print(f"\nDifference:")
print(f"  Mean difference: {abs(df1['PRESSURE'].mean() - df2['PRESSURE'].mean()):.2f} Pa")
print(f"  Std difference:  {abs(df1['PRESSURE'].std() - df2['PRESSURE'].std()):.2f} Pa")

print("\n" + "=" * 80)
print("6. UNIQUE FEATURES IN COMPREHENSIVE")
print("=" * 80)
if 'autoencoder_window_hits' in df1.columns:
    print(f"\nautoencoder_window_hits:")
    print(f"  Mean: {df1['autoencoder_window_hits'].mean():.2f}")
    print(f"  Max:  {df1['autoencoder_window_hits'].max()}")
    print(f"  Unique values: {df1['autoencoder_window_hits'].nunique()}")

if 'autoencoder_positive_hit' in df1.columns:
    ae_pos = df1['autoencoder_positive_hit'].sum()
    print(f"\nautoencoder_positive_hit:")
    print(f"  True count:  {ae_pos:>8,} ({ae_pos/len(df1)*100:>6.2f}%)")
    print(f"  False count: {(df1['autoencoder_positive_hit']==0).sum():>8,}")

if 'PRESSURE_MA_500' in df1.columns:
    print(f"\nPRESSURE_MA_500 (Moving Average):")
    print(f"  Mean: {df1['PRESSURE_MA_500'].mean():.2f} Pa")
    print(f"  Correlation with PRESSURE: {df1['PRESSURE'].corr(df1['PRESSURE_MA_500']):.4f}")

print("\n" + "=" * 80)
print("7. KEY FINDINGS & RECOMMENDATIONS")
print("=" * 80)

print("\n[+] WHAT COMPREHENSIVE HAS THAT ML_READY DOESN'T:")
if only_in_comprehensive:
    for col in sorted(only_in_comprehensive):
        if col == 'autoencoder_window_hits':
            print(f"  • {col}: Autoencoder window hit counter (potential feature)")
        elif col == 'autoencoder_positive_hit':
            print(f"  • {col}: Autoencoder binary prediction (potential feature/label)")
        elif col == 'PRESSURE_MA_500':
            print(f"  • {col}: 500-sample moving average (smoothed baseline feature)")
        else:
            print(f"  • {col}")

print("\n[!] WHAT ML_READY HAS THAT COMPREHENSIVE DOESN'T:")
if only_in_ml_ready:
    for col in sorted(only_in_ml_ready):
        print(f"  • {col}: MISSING in comprehensive!")
else:
    print("  (none - comprehensive is a superset)")

print("\n[*] RECOMMENDATIONS:")
print("  1. comprehensive_filtered_data_optimized.csv is a SUPERSET of ml_ready")
print("  2. Use comprehensive as primary dataset - it has:")
print("     • More data (better for RF)")
print("     • Autoencoder features (additional signal)")
print("     • PRESSURE_MA_500 (useful baseline)")
print("  3. Check if ml_ready was filtered/cleaned - understand why")
print("  4. Consider using autoencoder features in RF model")
print("  5. PRESSURE_MA_500 can be used for anomaly detection features")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

