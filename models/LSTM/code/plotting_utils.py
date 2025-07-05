# plotting_utils.py
"""
Plotting and visualization utilities for LSTM vortex detection.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_confidence_distribution(y_true, y_pred_proba, save_path='confidence_distribution.png'):
    """
    Plot distribution of confidence values for each class.
    """
    plt.figure(figsize=(12, 6))
    # Get confidence values for each class
    vortex_conf = y_pred_proba[y_true == 1]
    non_vortex_conf = y_pred_proba[y_true == 0]
    # Plot histograms
    plt.hist(vortex_conf, bins=50, alpha=0.5, label='Vortex', color='red')
    plt.hist(non_vortex_conf, bins=50, alpha=0.5, label='Non-Vortex', color='blue')
    plt.title('Distribution of Confidence Values')
    plt.xlabel('Confidence Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confidence_timeline(data, y_pred_proba, detection_windows, save_path='confidence_timeline.png'):
    """
    Plot confidence values over time with vortex events marked.
    """
    plt.figure(figsize=(15, 6))
    # Create time index
    time_index = range(len(y_pred_proba))
    # Plot confidence values
    plt.plot(time_index, y_pred_proba, label='Confidence', color='blue', alpha=0.7)
    # Mark detection windows
    for start, end in detection_windows:
        plt.axvspan(start, end, color='red', alpha=0.2, label='Detection Window' if start == detection_windows[0][0] else "")
    plt.title('Confidence Values Over Time')
    plt.xlabel('Time Index')
    plt.ylabel('Confidence Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Add stubs for any other plotting functions as needed 

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_detection_patterns(successful_patterns, failed_patterns, successful_sclks, failed_sclks,
                           min_slope_successful, min_slope_failed, short_window_slope_successful, 
                           short_window_slope_failed, sharp_drop_duration_successful, sharp_drop_duration_failed,
                           time_points, mean_successful, mean_failed, std_successful, std_failed):
    """Plot detection pattern analysis."""
    plt.figure(figsize=(22, 14))

    # 1. Mean patterns
    plt.subplot(3, 3, 1)
    plt.plot(time_points, mean_successful, label='Successful Detections', color='green', linewidth=2)
    plt.plot(time_points, mean_failed, label='False Alarms', color='red', linewidth=2)
    plt.fill_between(time_points, mean_successful - std_successful, mean_successful + std_successful, color='green', alpha=0.2)
    plt.fill_between(time_points, mean_failed - std_failed, mean_failed + std_failed, color='red', alpha=0.2)
    plt.title('Mean Pressure Patterns Around Detections')
    plt.xlabel('Time Points (relative to detection)')
    plt.ylabel('Pressure')
    plt.legend()
    plt.grid(True)
    
    # 2. Difference
    plt.subplot(3, 3, 2)
    plt.plot(time_points, mean_successful - mean_failed, color='blue', linewidth=2)
    plt.title('Difference (Successful - Failed)')
    plt.xlabel('Time Points (relative to detection)')
    plt.ylabel('Pressure Difference')
    plt.grid(True)
    
    # 3. Individual patterns
    plt.subplot(3, 3, 4)
    for i in range(min(30, len(successful_patterns))):
        plt.plot(time_points, successful_patterns[i], color='green', alpha=0.3)
    for i in range(min(30, len(failed_patterns))):
        plt.plot(time_points, failed_patterns[i], color='red', alpha=0.3)
    plt.title('Individual Patterns (First 30 of each)')
    plt.xlabel('Time Points (relative to detection)')
    plt.ylabel('Pressure')
    plt.grid(True)
    
    # 4. SCLK distribution
    plt.subplot(3, 3, 5)
    if len(successful_sclks) > 0:
        plt.hist(successful_sclks, bins=20, alpha=0.7, label='Successful', color='green')
    if len(failed_sclks) > 0:
        plt.hist(failed_sclks, bins=50, alpha=0.7, label='Failed', color='red')
    plt.title('SCLK Distribution of Detections')
    plt.xlabel('SCLK')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # 5. Minimum slope histogram
    plt.subplot(3, 3, 6)
    plt.hist(min_slope_successful, bins=30, alpha=0.7, label='Successful', color='green', range=(-1, 0.1))
    plt.hist(min_slope_failed, bins=30, alpha=0.7, label='Failed', color='red', range=(-1, 0.1))
    plt.title('Distribution of Minimum Slope in Detection Windows')
    plt.xlabel('Minimum Slope (sharpest drop)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # 6. Sharpest short-window slope (3-point drop)
    plt.subplot(3, 3, 7)
    plt.hist(short_window_slope_successful, bins=30, alpha=0.7, label='Successful', color='green', range=(-1, 0.1))
    plt.hist(short_window_slope_failed, bins=30, alpha=0.7, label='Failed', color='red', range=(-1, 0.1))
    plt.title('Sharpest 3-Point Drop in Detection Windows')
    plt.xlabel('Max 3-Point Drop')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)

    # 7. Drop duration
    plt.subplot(3, 3, 8)
    plt.hist(sharp_drop_duration_successful, bins=20, alpha=0.7, label='Successful', color='green')
    plt.hist(sharp_drop_duration_failed, bins=20, alpha=0.7, label='Failed', color='red')
    plt.title('Drop Duration in Detection Windows')
    plt.xlabel('Drop Duration (points)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('detection_patterns.png')
    plt.close()

def plot_confidence_analysis(successful_confidences, failed_confidences):
    """Plot confidence analysis."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(successful_confidences, bins=20, alpha=0.7, label='Successful', color='green', density=True)
    plt.hist(failed_confidences, bins=50, alpha=0.7, label='Failed', color='red', density=True)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # Plot confidence vs threshold analysis
    plt.subplot(1, 2, 2)
    thresholds = np.linspace(0.1, 0.9, 81)
    successful_counts = []
    failed_counts = []
    
    for threshold in thresholds:
        successful_counts.append(np.sum(successful_confidences >= threshold))
        failed_counts.append(np.sum(failed_confidences >= threshold))
    
    plt.plot(thresholds, successful_counts, label='Successful', color='green', linewidth=2)
    plt.plot(thresholds, failed_counts, label='Failed', color='red', linewidth=2)
    plt.title('Detections vs Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Number of Detections')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png')
    plt.close()

def plot_pressure_patterns(mean_vortex, mean_non_vortex, std_vortex, std_non_vortex, results_dir):
    """Plot pressure patterns around vortices vs non-vortices."""
    plt.figure(figsize=(12, 6))
    
    # Plot mean patterns
    plt.plot(mean_vortex, label='Vortex', color='red')
    plt.plot(mean_non_vortex, label='Non-Vortex', color='blue')
    
    # Plot standard deviation ranges
    plt.fill_between(range(len(mean_vortex)), 
                    mean_vortex - std_vortex, 
                    mean_vortex + std_vortex, 
                    color='red', alpha=0.2)
    plt.fill_between(range(len(mean_non_vortex)), 
                    mean_non_vortex - std_non_vortex, 
                    mean_non_vortex + std_non_vortex, 
                    color='blue', alpha=0.2)
    
    plt.title('Mean Pressure Patterns Around Vortices vs Non-Vortices')
    plt.xlabel('Time Steps')
    plt.ylabel('Pressure (Pa)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'pressure_patterns.png')
    plt.close()

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_continued_drop_analysis(
    continued_slope_successful, continued_slope_failed,
    total_drop_successful, total_drop_failed,
    consecutive_neg_successful, consecutive_neg_failed,
    drop_after_sharpest_successful, drop_after_sharpest_failed,
    total_drop_after_initial_successful, total_drop_after_initial_failed,
    lookahead,
    suffix=""
):
    """Plot continued drop analysis."""
    # 1. Avg Slope After Sharpest Drop (Next 5 pts)
    plt.figure(figsize=(8, 6))
    plt.hist(continued_slope_successful, bins=30, alpha=0.7, label='Successful', color='green')
    plt.hist(continued_slope_failed, bins=30, alpha=0.7, label='Failed', color='red')
    plt.title('Avg Slope After Sharpest Drop (Next 5 pts)')
    plt.xlabel('Avg Slope')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('continued_drop_avg_slope.png')
    plt.close()

    # 2. Total Drop Over Window
    plt.figure(figsize=(8, 6))
    plt.hist(total_drop_successful, bins=30, alpha=0.7, label='Successful', color='green')
    plt.hist(total_drop_failed, bins=30, alpha=0.7, label='Failed', color='red')
    plt.title('Total Drop Over Window')
    plt.xlabel('Total Drop (last - first)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('continued_drop_total_drop_window.png')
    plt.close()

    # 3. Consecutive Negative Slopes After Sharpest Drop
    plt.figure(figsize=(8, 6))
    plt.hist(consecutive_neg_successful, bins=range(lookahead+2), alpha=0.7, label='Successful', color='green', align='left')
    plt.hist(consecutive_neg_failed, bins=range(lookahead+2), alpha=0.7, label='Failed', color='red', align='left')
    plt.title('Consecutive Negative Slopes After Sharpest Drop')
    plt.xlabel('Consecutive Negative Slopes')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('continued_drop_consecutive_neg.png')
    plt.close()

    # 4. Total Drop After Sharpest Drop
    plt.figure(figsize=(8, 6))
    plt.hist(drop_after_sharpest_successful, bins=30, alpha=0.7, label='Successful', color='green')
    plt.hist(drop_after_sharpest_failed, bins=30, alpha=0.7, label='Failed', color='red')
    plt.title('Total Drop After Sharpest Drop')
    plt.xlabel('Total Drop (end - sharpest)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('continued_drop_after_sharpest.png')
    plt.close()

    # 5. Total Drop After Initial Drop
    plt.figure(figsize=(8, 6))
    plt.hist(total_drop_after_initial_successful, bins=30, alpha=0.7, label='Successful', color='green')
    plt.hist(total_drop_after_initial_failed, bins=30, alpha=0.7, label='Failed', color='red')
    plt.title('Total Drop After Initial Drop' + suffix)
    plt.xlabel('Total Drop (end - initial)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'continued_drop_analysis_total_drop_after_initial{suffix}.png')
    plt.close()

    # You can add more plots for other features as needed, e.g. continued_slope, drop_after_sharpest, etc.
    # Example for continued_slope:
    plt.figure(figsize=(10, 8))
    plt.hist(continued_slope_successful, bins=30, alpha=0.7, label='Successful', color='green')
    plt.hist(continued_slope_failed, bins=30, alpha=0.7, label='Failed', color='red')
    plt.title('Continued Slope After Sharpest Drop' + suffix)
    plt.xlabel(f'Avg Slope (next {lookahead} points)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'continued_drop_analysis_continued_slope{suffix}.png')
    plt.close()

    # Add more plots as needed for your analysis

def plot_sharpest_n_point_drops(successful_patterns, failed_patterns):
    """Plot sharpest N-point drop analysis."""
    window_sizes = [2, 3, 5, 10]
    sharpest_drops_successful = {}
    sharpest_drops_failed = {}
    for w in window_sizes:
        sharpest_drops_successful[w] = [np.min([p[i+w-1] - p[i] for i in range(len(p)-w+1)]) for p in successful_patterns]
        sharpest_drops_failed[w] = [np.min([p[i+w-1] - p[i] for i in range(len(p)-w+1)]) for p in failed_patterns]

    plt.figure(figsize=(18, 12))
    for idx, w in enumerate(window_sizes):
        plt.subplot(2, 2, idx+1)
        plt.hist(sharpest_drops_successful[w], bins=30, alpha=0.7, label='Successful', color='green', range=(-1, 0.1))
        plt.hist(sharpest_drops_failed[w], bins=30, alpha=0.7, label='Failed', color='red', range=(-1, 0.1))
        plt.title(f'Sharpest {w}-Point Drop in Detection Windows')
        plt.xlabel(f'Max {w}-Point Drop')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig('sharpest_n_point_drops.png')
    plt.close() 