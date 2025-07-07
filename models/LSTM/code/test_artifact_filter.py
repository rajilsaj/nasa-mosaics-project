"""
Quick test to evaluate artifact detector performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from artifact_detector import ArtifactDetector

def test_artifact_filter():
    """Test how the artifact detector filters data."""
    
    print("Testing artifact detector performance...")
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    
    # Split data
    n_samples = len(data)
    train_end = int(0.7 * n_samples)
    train_data = data.iloc[:train_end]
    
    print(f"Training data size: {len(train_data)}")
    
    # Initialize artifact detector
    artifact_detector = ArtifactDetector(window_size=60)
    
    # Test on a sample of data
    sample_size = min(10000, len(train_data))
    sample_data = train_data.iloc[:sample_size]
    
    print(f"Testing on sample of {sample_size} points...")
    
    # Count different types of windows
    window_size = 60
    total_windows = 0
    vortex_windows = 0
    artifact_windows = 0
    clean_windows = 0
    
    gt_detection = sample_data['gt_detection_win'].values
    gt_fwhm = sample_data['gt_fwhm'].values
    gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
    pressure_values = sample_data['PRESSURE'].values
    
    for i in range(window_size, len(sample_data)):
        total_windows += 1
        
        # Check if vortex
        window_contains_vortex = np.any(gt_combined[i-window_size:i] == 1)
        
        if window_contains_vortex:
            vortex_windows += 1
        else:
            # Check if artifact
            pressure_window = pressure_values[i-window_size:i].copy()
            artifact_info = artifact_detector._analyze_window_for_artifacts(pressure_window, i)
            
            if artifact_info['is_artifact']:
                artifact_windows += 1
            else:
                clean_windows += 1
    
    # Print results
    print(f"\nArtifact Detection Results:")
    print(f"Total windows: {total_windows}")
    print(f"Vortex windows: {vortex_windows} ({vortex_windows/total_windows*100:.1f}%)")
    print(f"Artifact windows: {artifact_windows} ({artifact_windows/total_windows*100:.1f}%)")
    print(f"Clean windows: {clean_windows} ({clean_windows/total_windows*100:.1f}%)")
    
    # Check if we have enough clean data
    if clean_windows < total_windows * 0.3:
        print(f"⚠️  WARNING: Only {clean_windows/total_windows*100:.1f}% clean data - might be too aggressive!")
    elif clean_windows > total_windows * 0.8:
        print(f"⚠️  WARNING: {clean_windows/total_windows*100:.1f}% clean data - might be too permissive!")
    else:
        print(f"✅ Good balance: {clean_windows/total_windows*100:.1f}% clean data")
    
    # Test performance
    import time
    start_time = time.time()
    
    # Test artifact detection speed
    test_windows = 1000
    for i in range(window_size, window_size + test_windows):
        pressure_window = pressure_values[i-window_size:i].copy()
        artifact_detector._analyze_window_for_artifacts(pressure_window, i)
    
    elapsed = time.time() - start_time
    windows_per_second = test_windows / elapsed
    
    print(f"\nPerformance Test:")
    print(f"Processed {test_windows} windows in {elapsed:.2f} seconds")
    print(f"Speed: {windows_per_second:.0f} windows/second")
    
    # Estimate full training time
    total_training_windows = len(train_data) - window_size
    estimated_time = total_training_windows / windows_per_second
    print(f"Estimated full training time: {estimated_time/60:.1f} minutes")
    
    if estimated_time > 30:
        print(f"⚠️  WARNING: Training will take {estimated_time/60:.1f} minutes - consider optimization!")

if __name__ == "__main__":
    test_artifact_filter() 