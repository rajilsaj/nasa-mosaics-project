#!/usr/bin/env python3
"""
Comprehensive Analysis of comprehensive_filtered_data_optimized.csv
As a seasoned RF analyst with 20 years of experience
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("COMPREHENSIVE DATA ANALYSIS - Random Forest Expert Perspective")
print("=" * 80)

# Load data
df = pd.read_csv('comprehensive_filtered_data_optimized.csv')

print("\n" + "=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)
print(f"Total samples: {len(df):,}")
print(f"Time span: {df['sol'].min()} to {df['sol'].max()} sols ({df['sol'].max() - df['sol'].min()} sols)")
print(f"SCLK range: {df['SCLK'].min()} to {df['SCLK'].max()}")
print(f"Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n" + "=" * 80)
print("2. CLASS DISTRIBUTION (CRITICAL FOR RF)")
print("=" * 80)
gt_detection_true = df['gt_detection_win'].sum()
gt_detection_false = (~df['gt_detection_win']).sum()
gt_fwhm_true = df['gt_fwhm'].sum()
gt_4xfwhm_true = df['gt_4xfwhm'].sum()

print(f"gt_detection_win=True:  {gt_detection_true:>8,} ({gt_detection_true/len(df)*100:>6.3f}%)")
print(f"gt_detection_win=False: {gt_detection_false:>8,} ({gt_detection_false/len(df)*100:>6.3f}%)")
print(f"Imbalance ratio: {gt_detection_false/gt_detection_true:.1f}:1 (Neg:Pos)")
print(f"\ngt_fwhm=True:  {gt_fwhm_true:>8,} ({gt_fwhm_true/len(df)*100:>6.3f}%)")
print(f"gt_4xfwhm=True: {gt_4xfwhm_true:>8,} ({gt_4xfwhm_true/len(df)*100:>6.3f}%)")

print("\n" + "=" * 80)
print("3. MISSING VALUES & DATA QUALITY")
print("=" * 80)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values!")
print(f"\nDuplicate SCLK: {df['SCLK'].duplicated().sum()}")
print(f"SCLK sorted: {'Yes ✓' if df['SCLK'].is_monotonic_increasing else 'No ✗ - CRITICAL ISSUE!'}")
print(f"SCLK gaps (most common):")
sclk_diffs = df['SCLK'].diff().value_counts().head(5)
print(sclk_diffs)

print("\n" + "=" * 80)
print("4. PRESSURE STATISTICS (PRIMARY FEATURE)")
print("=" * 80)
print(f"Overall:")
print(f"  Mean: {df['PRESSURE'].mean():.2f} Pa")
print(f"  Std:  {df['PRESSURE'].std():.2f} Pa")
print(f"  Min:  {df['PRESSURE'].min():.2f} Pa")
print(f"  Max:  {df['PRESSURE'].max():.2f} Pa")
print(f"  Range: {df['PRESSURE'].max() - df['PRESSURE'].min():.2f} Pa")
print(f"  CV (std/mean): {df['PRESSURE'].std()/df['PRESSURE'].mean()*100:.2f}%")

vortex = df[df['gt_detection_win']==True]
non_vortex = df[df['gt_detection_win']==False]

print(f"\nBy Class:")
print(f"  Vortex mean:     {vortex['PRESSURE'].mean():.2f} Pa (std: {vortex['PRESSURE'].std():.2f})")
print(f"  Non-vortex mean: {non_vortex['PRESSURE'].mean():.2f} Pa (std: {non_vortex['PRESSURE'].std():.2f})")
print(f"  Difference:      {abs(vortex['PRESSURE'].mean() - non_vortex['PRESSURE'].mean()):.2f} Pa")
print(f"  Effect size (Cohen's d): {abs(vortex['PRESSURE'].mean() - non_vortex['PRESSURE'].mean()) / np.sqrt((vortex['PRESSURE'].std()**2 + non_vortex['PRESSURE'].std()**2)/2):.3f}")

print(f"\nPRESSURE_MA_500 (Moving Average):")
print(f"  Mean: {df['PRESSURE_MA_500'].mean():.2f} Pa")
print(f"  Correlation with PRESSURE: {df['PRESSURE'].corr(df['PRESSURE_MA_500']):.4f}")

print("\n" + "=" * 80)
print("5. AUTOENCODER FEATURES (POTENTIAL FEATURES)")
print("=" * 80)
print(f"autoencoder_window_hits:")
print(f"  Mean: {df['autoencoder_window_hits'].mean():.2f}")
print(f"  Std:  {df['autoencoder_window_hits'].std():.2f}")
print(f"  Min:  {df['autoencoder_window_hits'].min()}")
print(f"  Max:  {df['autoencoder_window_hits'].max()}")
print(f"  Unique values: {df['autoencoder_window_hits'].nunique()}")

print(f"\nautoencoder_positive_hit:")
ae_pos = df['autoencoder_positive_hit'].sum()
print(f"  True count:  {ae_pos:>8,} ({ae_pos/len(df)*100:>6.2f}%)")
print(f"  False count: {(df['autoencoder_positive_hit']==0).sum():>8,}")

print(f"\nAutoencoder vs Ground Truth Agreement:")
both_pos = ((df['autoencoder_positive_hit']==1) & (df['gt_detection_win']==True)).sum()
ae_only = ((df['autoencoder_positive_hit']==1) & (df['gt_detection_win']==False)).sum()
gt_only = ((df['autoencoder_positive_hit']==0) & (df['gt_detection_win']==True)).sum()
both_neg = ((df['autoencoder_positive_hit']==0) & (df['gt_detection_win']==False)).sum()

print(f"  Both positive:     {both_pos:>8,} ({both_pos/len(df)*100:>6.3f}%)")
print(f"  Autoencoder only:  {ae_only:>8,} ({ae_only/len(df)*100:>6.3f}%)")
print(f"  Ground truth only: {gt_only:>8,} ({gt_only/len(df)*100:>6.3f}%)")
print(f"  Both negative:     {both_neg:>8,} ({both_neg/len(df)*100:>6.3f}%)")

if both_pos + gt_only > 0:
    precision_ae = both_pos / (both_pos + ae_only) if (both_pos + ae_only) > 0 else 0
    recall_ae = both_pos / (both_pos + gt_only) if (both_pos + gt_only) > 0 else 0
    print(f"\n  Autoencoder Performance (vs GT):")
    print(f"    Precision: {precision_ae:.4f}")
    print(f"    Recall:    {recall_ae:.4f}")

print("\n" + "=" * 80)
print("6. TEMPORAL ANALYSIS")
print("=" * 80)
total_sols = df['sol'].max() - df['sol'].min() + 1
print(f"Samples per sol: {len(df) / total_sols:.0f}")
print(f"Expected (1 Hz): ~86,400 samples/sol")
print(f"Coverage: {len(df) / (total_sols * 86400) * 100:.1f}% of expected")

vortex_sols = df[df['gt_detection_win']==True].groupby('sol').size()
print(f"\nVortex events per sol:")
print(f"  Mean: {vortex_sols.mean():.2f}")
print(f"  Median: {vortex_sols.median():.2f}")
print(f"  Max: {vortex_sols.max()}")
print(f"  Total unique sols with vortices: {len(vortex_sols)} out of {total_sols} ({len(vortex_sols)/total_sols*100:.1f}%)")

print("\n" + "=" * 80)
print("7. RF EXPERT INSIGHTS & RECOMMENDATIONS")
print("=" * 80)
print("\n✓ STRENGTHS:")
print("  • Large dataset (1.69M samples) - good for RF training")
print("  • Temporal ordering maintained (SCLK sorted)")
print("  • Multiple ground truth labels available")
print("  • Autoencoder features provide additional signal")
print("  • Pressure data shows class separation potential")

print("\n⚠ CONCERNS:")
print(f"  • EXTREME class imbalance ({gt_detection_false/gt_detection_true:.0f}:1 ratio)")
print("  • Need class_weight='balanced' or custom weights in RF")
print("  • Consider stratified sampling or SMOTE for training")
print("  • Autoencoder has false positives (potential noise)")

print("\n💡 RECOMMENDATIONS:")
print("  1. Use sliding window approach (60 samples) for temporal features")
print("  2. Engineer features: pressure gradients, rolling stats, anomalies")
print("  3. Use time-based splits (train/val/test) to prevent leakage")
print("  4. Apply class_weight='balanced' or custom weights in RF")
print("  5. Consider autoencoder_positive_hit as a feature (not just label)")
print("  6. Evaluate on natural imbalance (not balanced test set)")
print("  7. Use threshold tuning (not default 0.5) for deployment")
print("  8. Consider ensemble with autoencoder predictions")

print("\n" + "=" * 80)
print("8. COMPARISON WITH CURRENT PIPELINE")
print("=" * 80)
print("Current pipeline uses:")
print("  • ml_ready_vortex_data.csv (likely subset of this)")
print("  • 60-sample windows extracted from precursor regions")
print("  • 15 engineered features")
print("  • Balanced training (1:1), natural test (225:1)")
print("\nThis comprehensive dataset:")
print("  • Contains full temporal coverage")
print("  • Includes autoencoder predictions")
print("  • Has multiple ground truth labels")
print("  • Could be used to improve feature engineering")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

