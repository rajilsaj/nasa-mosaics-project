#!/usr/bin/env python3
"""
Pre-compute Features for Sliding Windows
========================================

This script pre-computes features for sliding windows and saves them to CSV.
Run this ONCE, then use the saved features for fast evaluation.

Usage:
    python precompute_sliding_features.py --split val
    python precompute_sliding_features.py --split test
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_slope(x, y):
    """Calculate linear regression slope."""
    if len(x) < 2:
        return 0.0
    try:
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    except:
        return 0.0

def engineer_features_for_window(window_data, global_mean, global_std):
    """Engineer 15 features from a 60-sample pressure window."""
    if window_data is None or len(window_data) == 0:
        return {f'feature_{i}': 0.0 for i in range(15)}
    
    # Handle both uppercase and lowercase column names
    if 'PRESSURE' in window_data.columns:
        pressure = window_data['PRESSURE'].values
    elif 'pressure' in window_data.columns:
        pressure = window_data['pressure'].values
    else:
        return {f'feature_{i}': 0.0 for i in range(15)}
    
    n_samples = len(pressure)
    
    if n_samples == 0:
        return {f'feature_{i}': 0.0 for i in range(15)}
    
    # Ensure we have at least 60 samples (pad with last value if needed)
    if n_samples < 60:
        padding = np.full(60 - n_samples, pressure[-1])
        pressure = np.concatenate([pressure, padding])
        n_samples = 60
    
    # Calculate indices for first and second half
    mid_point = n_samples // 2
    first_half = pressure[:mid_point]
    second_half = pressure[mid_point:]
    
    # Feature 1: Overall slope
    x = np.arange(n_samples)
    overall_slope = calculate_slope(x, pressure)
    
    # Feature 2: First half slope
    x_first = np.arange(len(first_half))
    first_half_slope = calculate_slope(x_first, first_half)
    
    # Feature 3: Second half slope
    x_second = np.arange(len(second_half))
    second_half_slope = calculate_slope(x_second, second_half)
    
    # Feature 4: Trend consistency
    if len(pressure) >= 4:
        window_size = min(10, len(pressure) // 3)
        slopes = []
        for i in range(len(pressure) - window_size):
            x = np.arange(window_size)
            y = pressure[i:i + window_size]
            slope = calculate_slope(x, y)
            slopes.append(slope)
        slope_std = np.std(slopes) if slopes else 0
        trend_consistency = 1.0 / (1.0 + slope_std) if slope_std > 0 else 1.0
    else:
        trend_consistency = 0.0
    
    # Feature 5: Pressure drop
    pressure_drop = np.max(pressure) - np.min(pressure)
    
    # Feature 6: Drop rate
    max_drop = 0.0
    for i in range(len(pressure) - 1):
        drop = pressure[i] - pressure[i + 1]
        max_drop = max(max_drop, drop)
    drop_rate = max_drop
    
    # Feature 7: Minimum position
    min_idx = np.argmin(pressure)
    min_position = min_idx / (len(pressure) - 1) if len(pressure) > 1 else 0.5
    
    # Feature 8: Mean
    mean = np.mean(pressure)
    
    # Feature 9: Standard deviation
    std = np.std(pressure)
    
    # Feature 10: Range
    range_val = np.max(pressure) - np.min(pressure)
    
    # Feature 11: First half mean
    first_half_mean = np.mean(first_half)
    
    # Feature 12: Second half mean
    second_half_mean = np.mean(second_half)
    
    # Feature 13: Mean ratio
    mean_ratio = second_half_mean / first_half_mean if first_half_mean != 0 else 1.0
    
    # Feature 14: Minimum z-score
    min_pressure = np.min(pressure)
    if global_std > 0:
        min_zscore = (min_pressure - global_mean) / global_std
    else:
        min_zscore = 0.0
    
    # Feature 15: Anomaly strength
    if len(pressure) >= 3:
        x = np.arange(len(pressure))
        slope, intercept, _, _, _ = stats.linregress(x, pressure)
        min_idx = np.argmin(pressure)
        expected_pressure = slope * min_idx + intercept
        actual_pressure = pressure[min_idx]
        anomaly_strength = abs(actual_pressure - expected_pressure)
    else:
        anomaly_strength = 0.0
    
    features = {
        'overall_slope': overall_slope,
        'first_half_slope': first_half_slope,
        'second_half_slope': second_half_slope,
        'trend_consistency': trend_consistency,
        'pressure_drop': pressure_drop,
        'drop_rate': drop_rate,
        'min_position': min_position,
        'mean': mean,
        'std': std,
        'range': range_val,
        'first_half_mean': first_half_mean,
        'second_half_mean': second_half_mean,
        'mean_ratio': mean_ratio,
        'min_zscore': min_zscore,
        'anomaly_strength': anomaly_strength
    }
    
    return features

def main():
    parser = argparse.ArgumentParser(description='Pre-compute features for sliding windows')
    parser.add_argument('--split', choices=['val', 'test'], required=True, help='Which split to process')
    parser.add_argument('--step_size', type=int, default=10, help='Step size (default: 10)')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"PRE-COMPUTING FEATURES FOR {args.split.upper()} SLIDING WINDOWS")
    print("=" * 70)
    
    # Load sliding windows
    input_file = f"{args.split}_sliding_windows_step{args.step_size}.csv"
    output_file = f"{args.split}_sliding_features.csv"
    
    print(f"\nLoading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} sliding windows")
    
    # Calculate global statistics
    print("\nCalculating global statistics...")
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    all_pressures = []
    
    for _, row in sample_df.iterrows():
        try:
            window_data = pd.read_json(row['window_data'], orient='records')
            if 'PRESSURE' in window_data.columns:
                all_pressures.extend(window_data['PRESSURE'].values)
        except:
            continue
    
    global_mean = np.mean(all_pressures) if all_pressures else 745.0
    global_std = np.std(all_pressures) if all_pressures else 8.0
    print(f"  Global mean: {global_mean:.3f}")
    print(f"  Global std: {global_std:.3f}")
    
    # Engineer features for all windows
    print(f"\nEngineering features for {len(df):,} windows...")
    all_features = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        try:
            window_data = pd.read_json(row['window_data'], orient='records')
            features = engineer_features_for_window(window_data, global_mean, global_std)
            
            feature_row = {
                'window_id': row['window_id'],
                'start_idx': row['start_idx'],
                'end_idx': row['end_idx'],
                'start_sclk': row['start_sclk'],
                'end_sclk': row['end_sclk'],
                'label': row['label'],
                **features
            }
            all_features.append(feature_row)
        except:
            continue
    
    # Save to CSV
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_file, index=False)
    
    print(f"\n[SUCCESS] Saved {len(features_df):,} feature vectors to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.1f} MB")
    
    # Show label distribution
    label_counts = features_df['label'].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(features_df)) * 100
        print(f"  {label}: {count:,} ({percentage:.1f}%)")
    
    print("\n" + "=" * 70)
    print("PRE-COMPUTATION COMPLETED!")
    print("=" * 70)
    print(f"\nNext step: Use '{output_file}' for fast evaluation!")

if __name__ == "__main__":
    main()



