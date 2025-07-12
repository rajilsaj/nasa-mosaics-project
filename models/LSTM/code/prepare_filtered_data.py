#!/usr/bin/env python3
"""
Prepare filtered data from autoencoder for separate classifier training.

This script loads a trained autoencoder, filters the data, and saves
the filtered sequences for the classifier to train on.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
import pickle
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import the autoencoder
from lstm_autoencoder import VortexLSTMAutoencoder

def prepare_filtered_data(data_path, autoencoder_path, output_path, window_size=60, threshold_percentile=95, debug=False):
    """
    Prepare filtered data from autoencoder for classifier training.
    
    Args:
        data_path: Path to input data CSV
        autoencoder_path: Path to trained autoencoder model
        output_path: Path to save filtered data
        window_size: Window size for sequences
        threshold_percentile: Percentile for autoencoder threshold (default: 95)
        debug: Enable debug output
    """
    
    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} samples")
    
    # Initialize autoencoder
    autoencoder = VortexLSTMAutoencoder(window_size=window_size, debug=debug)
    
    # Load trained autoencoder
    print(f"Loading autoencoder from: {autoencoder_path}")
    try:
        # Try loading with custom objects for compatibility
        from tensorflow.keras.metrics import MeanSquaredError
        autoencoder.model = tf.keras.models.load_model(
            autoencoder_path, 
            custom_objects={'mse': MeanSquaredError()}
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        # Try loading with compile=False
        autoencoder.model = tf.keras.models.load_model(
            autoencoder_path, 
            compile=False
        )
    
    # Set threshold using normal data
    print("Setting autoencoder threshold...")
    print("Preparing normal sequences for threshold calculation...")
    normal_sequences = autoencoder.prepare_normal_sequences(data)
    
    print(f"Computing reconstruction errors for {len(normal_sequences):,} normal sequences...")
    # Compute reconstruction errors with progress bar
    reconstruction_errors = []
    batch_size = 1000  # Process in batches to show progress
    
    for i in tqdm(range(0, len(normal_sequences), batch_size), desc="Computing reconstruction errors"):
        batch = normal_sequences[i:i+batch_size]
        batch_errors = autoencoder.predict_reconstruction_error(batch)
        reconstruction_errors.extend(batch_errors)
    
    reconstruction_errors = np.array(reconstruction_errors)
    autoencoder.reconstruction_threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    print(f"Autoencoder threshold set to: {autoencoder.reconstruction_threshold:.6f}")
    
    # Prepare all sequences
    print("\nPreparing all sequences...")
    sequences, labels = autoencoder.prepare_test_sequences(data)
    
    # Calculate initial statistics
    total_sequences = len(sequences)
    total_vortex = sum(labels)
    total_normal = total_sequences - total_vortex
    
    print(f"\n=== INITIAL DATA STATISTICS ===")
    print(f"Total sequences: {total_sequences:,}")
    print(f"Vortex sequences: {total_vortex:,} ({total_vortex/total_sequences*100:.1f}%)")
    print(f"Normal sequences: {total_normal:,} ({total_normal/total_sequences*100:.1f}%)")
    
    # Filter sequences with autoencoder
    print("\n=== FILTERING WITH AUTOENCODER ===")
    print("Computing reconstruction errors for all sequences...")
    
    # Compute reconstruction errors with progress bar
    reconstruction_errors = []
    batch_size = 1000  # Process in batches to show progress
    
    for i in tqdm(range(0, len(sequences), batch_size), desc="Computing reconstruction errors"):
        batch = sequences[i:i+batch_size]
        batch_errors = autoencoder.predict_reconstruction_error(batch)
        reconstruction_errors.extend(batch_errors)
    
    reconstruction_errors = np.array(reconstruction_errors)
    anomalies = reconstruction_errors > autoencoder.reconstruction_threshold
    
    # Calculate filtering statistics
    filtered_sequences = sequences[anomalies]
    filtered_labels = labels[anomalies]
    filtered_indices = np.where(anomalies)[0]
    
    filtered_total = len(filtered_sequences)
    filtered_vortex = sum(filtered_labels)
    filtered_normal = filtered_total - filtered_vortex
    
    # Calculate filtering rates
    vortex_retention_rate = filtered_vortex / total_vortex if total_vortex > 0 else 0
    normal_filtering_rate = (total_normal - filtered_normal) / total_normal if total_normal > 0 else 0
    overall_filtering_rate = (total_sequences - filtered_total) / total_sequences
    
    print(f"Autoencoder threshold: {autoencoder.reconstruction_threshold:.6f}")
    print(f"Sequences above threshold (anomalies): {sum(anomalies):,}")
    print(f"Sequences below threshold (normal): {sum(~anomalies):,}")
    
    print(f"\n=== FILTERING RESULTS ===")
    print(f"Filtered total sequences: {filtered_total:,} ({filtered_total/total_sequences*100:.1f}% of original)")
    print(f"Filtered vortex sequences: {filtered_vortex:,} ({filtered_vortex/total_vortex*100:.1f}% of original vortices)")
    print(f"Filtered normal sequences: {filtered_normal:,} ({filtered_normal/total_normal*100:.1f}% of original normal)")
    
    print(f"\n=== FILTERING EFFICIENCY ===")
    print(f"Vortex retention rate: {vortex_retention_rate*100:.1f}% (vortices kept)")
    print(f"Normal filtering rate: {normal_filtering_rate*100:.1f}% (normal removed)")
    print(f"Overall data reduction: {overall_filtering_rate*100:.1f}% (total removed)")
    
    # Calculate class balance changes
    original_vortex_ratio = total_vortex / total_sequences
    filtered_vortex_ratio = filtered_vortex / filtered_total if filtered_total > 0 else 0
    
    print(f"\n=== CLASS BALANCE ANALYSIS ===")
    print(f"Original vortex ratio: {original_vortex_ratio*100:.1f}%")
    print(f"Filtered vortex ratio: {filtered_vortex_ratio*100:.1f}%")
    print(f"Balance improvement: {(filtered_vortex_ratio/original_vortex_ratio-1)*100:+.1f}%")
    
    # Save filtered data
    filtered_data = {
        'sequences': filtered_sequences,
        'labels': filtered_labels,
        'original_indices': filtered_indices,
        'filter_ratio': filtered_total/total_sequences,
        'vortex_retention_rate': vortex_retention_rate,
        'normal_filtering_rate': normal_filtering_rate,
        'window_size': window_size,
        'autoencoder_threshold': autoencoder.reconstruction_threshold,
        'filtering_stats': {
            'total_original': total_sequences,
            'total_filtered': filtered_total,
            'vortex_original': total_vortex,
            'vortex_filtered': filtered_vortex,
            'normal_original': total_normal,
            'normal_filtered': filtered_normal
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(filtered_data, f)
    
    print(f"\nFiltered data saved to: {output_path}")
    
    return filtered_data

def main():
    """Main function to prepare filtered data."""
    parser = argparse.ArgumentParser(description='Prepare filtered data from autoencoder')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--autoencoder_path', type=str, default='autoencoder_model.h5', help='Path to trained autoencoder')
    parser.add_argument('--output_path', type=str, default='filtered_data.pkl', help='Path to save filtered data')
    parser.add_argument('--window_size', type=int, default=60, help='Window size for sequences')
    parser.add_argument('--threshold_percentile', type=float, default=95, help='Autoencoder threshold percentile (default: 95)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Check if autoencoder exists
    if not Path(args.autoencoder_path).exists():
        print(f"Error: Autoencoder not found at {args.autoencoder_path}")
        print("Please train the autoencoder first:")
        print(f"python lstm_autoencoder.py --retrain")
        return
    
    # Prepare filtered data
    filtered_data = prepare_filtered_data(
        data_path=args.data_path,
        autoencoder_path=args.autoencoder_path,
        output_path=args.output_path,
        window_size=args.window_size,
        threshold_percentile=args.threshold_percentile,
        debug=args.debug
    )
    
    print(f"\n=== NEXT STEPS ===")
    print(f"Now you can train the classifier on this filtered data:")
    print(f"python classifier_model.py --filtered_data {args.output_path} --retrain")

if __name__ == "__main__":
    main() 