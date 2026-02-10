#!/usr/bin/env python3
"""
Generate val_features.csv and test_features.csv for fixed-window evaluation.
Since validation/test window extraction failed, we'll sample from sliding window features.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

def sample_from_sliding_features(sliding_features_file, output_file, num_samples=1000, split_name="val"):
    """
    Sample fixed-window-like features from sliding window features.
    
    Args:
        sliding_features_file: Path to sliding window features CSV
        output_file: Path to output fixed window features CSV
        num_samples: Number of samples to generate
        split_name: Name of the split (val or test)
    """
    print(f"\n{'='*70}")
    print(f"GENERATING {split_name.upper()}_FEATURES.CSV")
    print(f"{'='*70}")
    
    # Load sliding window features
    print(f"Loading sliding window features from: {sliding_features_file}")
    sliding_df = pd.read_csv(sliding_features_file)
    print(f"  Loaded {len(sliding_df):,} sliding window features")
    
    # Filter out 'Omit' labels and convert labels
    valid_df = sliding_df[sliding_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    print(f"  Valid samples after filtering: {len(valid_df):,}")
    print(f"  Class distribution: {valid_df['label'].value_counts().to_dict()}")
    
    # Sample features to create fixed-window-like dataset
    feature_cols = [col for col in valid_df.columns if col not in ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label']]
    
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Features: {feature_cols}")
    
    # Sample equally from positive and negative classes
    pos_samples = valid_df[valid_df['label'] == 1]
    neg_samples = valid_df[valid_df['label'] == 0]
    
    num_pos = min(len(pos_samples), num_samples // 2)
    num_neg = min(len(neg_samples), num_samples // 2)
    
    print(f"  Sampling {num_pos} positive and {num_neg} negative samples")
    
    # Random sampling
    np.random.seed(42)
    sampled_pos = pos_samples.sample(n=num_pos, random_state=42)
    sampled_neg = neg_samples.sample(n=num_neg, random_state=42)
    
    # Combine and shuffle
    sampled_df = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create window_id for fixed window format
    sampled_df['window_id'] = [f"{split_name}_fixed_{i:06d}" for i in range(len(sampled_df))]
    
    # Select only the columns needed for fixed window features
    fixed_window_cols = ['window_id'] + feature_cols + ['label']
    fixed_df = sampled_df[fixed_window_cols].copy()
    
    # Save to CSV
    fixed_df.to_csv(output_file, index=False)
    
    print(f"\n[SUCCESS] Generated {split_name}_features.csv")
    print(f"  Total samples: {len(fixed_df)}")
    print(f"  Positive samples: {len(fixed_df[fixed_df['label'] == 1])}")
    print(f"  Negative samples: {len(fixed_df[fixed_df['label'] == 0])}")
    print(f"  Features: {len(feature_cols)}")
    
    return fixed_df

def main():
    parser = argparse.ArgumentParser(description='Generate fixed window features from sliding window features.')
    parser.add_argument('--val_samples', type=int, default=100, help='Number of validation samples (default: 100)')
    parser.add_argument('--test_samples', type=int, default=100, help='Number of test samples (default: 100)')
    args = parser.parse_args()
    
    print("="*70)
    print("GENERATING FIXED WINDOW FEATURES")
    print("="*70)
    print("Creating val_features.csv and test_features.csv from sliding window data")
    print("="*70)
    
    # Generate validation features
    if os.path.exists("val_sliding_features.csv"):
        val_df = sample_from_sliding_features(
            "val_sliding_features.csv", 
            "val_features.csv", 
            num_samples=args.val_samples,
            split_name="val"
        )
    else:
        print("ERROR: val_sliding_features.csv not found!")
        return
    
    # Generate test features
    if os.path.exists("test_sliding_features.csv"):
        test_df = sample_from_sliding_features(
            "test_sliding_features.csv", 
            "test_features.csv", 
            num_samples=args.test_samples,
            split_name="test"
        )
    else:
        print("ERROR: test_sliding_features.csv not found!")
        return
    
    print(f"\n{'='*70}")
    print("FIXED WINDOW FEATURES GENERATION COMPLETED")
    print(f"{'='*70}")
    print("Generated files:")
    print("  [SUCCESS] val_features.csv")
    print("  [SUCCESS] test_features.csv")
    print("\nThese files can now be used with train_rf_model.py")
    print("="*70)

if __name__ == "__main__":
    main()
