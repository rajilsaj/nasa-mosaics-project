import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter

def calculate_rate_of_change(values, window_size):
    """Calculate rate of change using a centered window."""
    time_delta = window_size // 2
    rate = np.zeros_like(values, dtype=float)
    
    for i in range(len(values)):
        start_idx = max(0, i - time_delta)
        end_idx = min(len(values), i + time_delta + 1)
        if end_idx - start_idx < 2:  # Need at least 2 points
            continue
        time_points = np.arange(end_idx - start_idx)
        coeffs = np.polyfit(time_points, values[start_idx:end_idx], 1)
        rate[i] = coeffs[0]  # First coefficient is the slope
    
    return rate

def find_continuous_regions(binary_array):
    """Find start and end indices of continuous regions of 1s in a binary array."""
    padded = np.concatenate([[0], binary_array, [0]])
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))

def plot_background_regions(ax, data):
    """Plot the background regions for vortex windows."""
    regions_4xfwhm = find_continuous_regions(data['gt_4xfwhm'].values)
    regions_detection = find_continuous_regions(data['gt_detection_win'].values)
    regions_fwhm = find_continuous_regions(data['gt_fwhm'].values)
    
    for start, end in regions_4xfwhm:
        ax.axvspan(start, end, color='gray', alpha=0.3)
    for start, end in regions_detection:
        ax.axvspan(start, end, color='red', alpha=0.3)
    for start, end in regions_fwhm:
        ax.axvspan(start, end, color='green', alpha=0.3)

def compare_pressure_changes():
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    pressure_values = data['PRESSURE'].values
    
    # Calculate different types of pressure changes
    simple_diff = np.diff(pressure_values, prepend=pressure_values[0])
    
    # Rate of change over different windows
    roc_small = calculate_rate_of_change(pressure_values, window_size=5)  # Short window
    roc_medium = calculate_rate_of_change(pressure_values, window_size=11)  # Medium window
    roc_large = calculate_rate_of_change(pressure_values, window_size=21)  # Large window
    
    # Smooth the rate of change signals
    roc_small_smooth = savgol_filter(roc_small, window_length=5, polyorder=2)
    roc_medium_smooth = savgol_filter(roc_medium, window_length=11, polyorder=2)
    roc_large_smooth = savgol_filter(roc_large, window_length=21, polyorder=2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    
    # Plot raw pressure
    axes[0].scatter(data.index, pressure_values, c='black', s=1, alpha=0.6)
    axes[0].set_ylabel('Pressure (Pa)')
    axes[0].set_title('Raw Pressure Values')
    plot_background_regions(axes[0], data)
    
    # Plot simple differences
    axes[1].scatter(data.index, simple_diff, c='blue', s=1, alpha=0.6)
    axes[1].set_ylabel('Pressure\nDifference')
    axes[1].set_title('Simple Point-to-Point Differences')
    plot_background_regions(axes[1], data)
    
    # Plot different window sizes of rate of change
    axes[2].scatter(data.index, roc_small_smooth, c='purple', s=1, alpha=0.6)
    axes[2].set_ylabel('Rate of Change\n(Pa/sample)')
    axes[2].set_title('Rate of Change - Small Window (5 samples)')
    plot_background_regions(axes[2], data)
    
    axes[3].scatter(data.index, roc_medium_smooth, c='green', s=1, alpha=0.6)
    axes[3].set_ylabel('Rate of Change\n(Pa/sample)')
    axes[3].set_title('Rate of Change - Medium Window (11 samples)')
    plot_background_regions(axes[3], data)
    
    axes[4].scatter(data.index, roc_large_smooth, c='red', s=1, alpha=0.6)
    axes[4].set_ylabel('Rate of Change\n(Pa/sample)')
    axes[4].set_title('Rate of Change - Large Window (21 samples)')
    plot_background_regions(axes[4], data)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.3, label='gt_4xfwhm'),
        Patch(facecolor='red', alpha=0.3, label='gt_detection_win'),
        Patch(facecolor='green', alpha=0.3, label='gt_fwhm')
    ]
    for ax in axes:
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Set x-axis label
    axes[-1].set_xlabel('SCLK')
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = Path(__file__).parent.parent / 'results' / 'pressure_change_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Calculate and print statistics for each method during vortex vs non-vortex periods
    methods = {
        'Simple Differences': simple_diff,
        'RoC Small Window': roc_small_smooth,
        'RoC Medium Window': roc_medium_smooth,
        'RoC Large Window': roc_large_smooth
    }
    
    print("\nSignal-to-Noise Analysis:")
    for name, values in methods.items():
        vortex_vals = np.abs(values[data['gt_fwhm'] == 1])
        non_vortex_vals = np.abs(values[data['gt_fwhm'] == 0])
        
        vortex_mean = np.mean(vortex_vals)
        non_vortex_mean = np.mean(non_vortex_vals)
        ratio = vortex_mean / non_vortex_mean
        
        print(f"\n{name}:")
        print(f"Vortex Mean Magnitude: {vortex_mean:.6f}")
        print(f"Non-Vortex Mean Magnitude: {non_vortex_mean:.6f}")
        print(f"Signal-to-Noise Ratio: {ratio:.2f}x")
        
        # Calculate standard deviation of non-vortex periods (noise level)
        noise_std = np.std(non_vortex_vals)
        print(f"Noise StdDev: {noise_std:.6f}")
        print(f"Signal-to-Noise StdDev Ratio: {vortex_mean/noise_std:.2f}x")

if __name__ == "__main__":
    compare_pressure_changes() 