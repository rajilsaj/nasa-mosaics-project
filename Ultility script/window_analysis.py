#!/usr/bin/env python3
"""
Comprehensive Window-Level Analysis for Mars Vortex Detection
Analyzes the raw 60-sample windows to understand temporal patterns and characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_window_data():
    """Load raw window data for analysis."""
    print("Loading window data...")
    
    # Load balanced training data (contains raw window samples)
    df = pd.read_csv('train_balanced.csv')
    print(f"Total window samples: {len(df):,} rows")
    print(f"Unique windows: {df['window_id'].nunique()}")
    print(f"Window size: 60 samples per window")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return df

def analyze_window_statistics(df):
    """Analyze basic statistics of windows."""
    print("\n" + "="*60)
    print("WINDOW-LEVEL STATISTICAL ANALYSIS")
    print("="*60)
    
    # Group by window_id and label
    window_stats = df.groupby(['window_id', 'label']).agg({
        'PRESSURE': ['mean', 'std', 'min', 'max', 'count'],
        'SCLK': ['min', 'max']
    }).round(4)
    
    window_stats.columns = ['pressure_mean', 'pressure_std', 'pressure_min', 'pressure_max', 'sample_count', 'sclk_start', 'sclk_end']
    window_stats = window_stats.reset_index()
    
    print(f"Window statistics computed for {len(window_stats)} windows")
    print(f"Average pressure range per window:")
    print(f"  Vortex windows: {window_stats[window_stats['label']==1]['pressure_max'].mean():.3f} - {window_stats[window_stats['label']==1]['pressure_min'].mean():.3f}")
    print(f"  Non-vortex windows: {window_stats[window_stats['label']==0]['pressure_max'].mean():.3f} - {window_stats[window_stats['label']==0]['pressure_min'].mean():.3f}")
    
    return window_stats

def create_window_trajectory_plots(df, save_path='window_analysis_trajectories.png'):
    """Create plots showing pressure trajectories for sample windows."""
    print("\nCreating window trajectory plots...")
    
    # Sample windows from each class
    vortex_windows = df[df['label'] == 1]['window_id'].unique()[:10]
    non_vortex_windows = df[df['label'] == 0]['window_id'].unique()[:10]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Sample vortex windows
    ax1 = axes[0, 0]
    for i, window_id in enumerate(vortex_windows):
        window_data = df[df['window_id'] == window_id].sort_values('SCLK')
        ax1.plot(range(len(window_data)), window_data['PRESSURE'], 
                alpha=0.7, linewidth=1, label=f'Vortex {i+1}' if i < 3 else "")
    ax1.set_title('Sample Vortex Windows (Pressure Trajectories)')
    ax1.set_xlabel('Sample Index (0-59)')
    ax1.set_ylabel('Pressure (Pa)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Sample non-vortex windows
    ax2 = axes[0, 1]
    for i, window_id in enumerate(non_vortex_windows):
        window_data = df[df['window_id'] == window_id].sort_values('SCLK')
        ax2.plot(range(len(window_data)), window_data['PRESSURE'], 
                alpha=0.7, linewidth=1, label=f'Non-Vortex {i+1}' if i < 3 else "")
    ax2.set_title('Sample Non-Vortex Windows (Pressure Trajectories)')
    ax2.set_xlabel('Sample Index (0-59)')
    ax2.set_ylabel('Pressure (Pa)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Average trajectories
    ax3 = axes[1, 0]
    vortex_avg = []
    non_vortex_avg = []
    
    for i in range(60):  # 60 samples per window
        vortex_samples = df[(df['label'] == 1) & (df.groupby('window_id').cumcount() == i)]['PRESSURE']
        non_vortex_samples = df[(df['label'] == 0) & (df.groupby('window_id').cumcount() == i)]['PRESSURE']
        
        if len(vortex_samples) > 0:
            vortex_avg.append(vortex_samples.mean())
        else:
            vortex_avg.append(np.nan)
            
        if len(non_vortex_samples) > 0:
            non_vortex_avg.append(non_vortex_samples.mean())
        else:
            non_vortex_avg.append(np.nan)
    
    ax3.plot(range(60), vortex_avg, 'r-', linewidth=2, label='Vortex Average')
    ax3.plot(range(60), non_vortex_avg, 'b-', linewidth=2, label='Non-Vortex Average')
    ax3.fill_between(range(60), vortex_avg, non_vortex_avg, alpha=0.3, color='gray')
    ax3.set_title('Average Pressure Trajectories by Class')
    ax3.set_xlabel('Sample Index (0-59)')
    ax3.set_ylabel('Average Pressure (Pa)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pressure difference (vortex - non_vortex)
    ax4 = axes[1, 1]
    pressure_diff = np.array(vortex_avg) - np.array(non_vortex_avg)
    ax4.plot(range(60), pressure_diff, 'g-', linewidth=2, label='Pressure Difference')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_title('Pressure Difference (Vortex - Non-Vortex)')
    ax4.set_xlabel('Sample Index (0-59)')
    ax4.set_ylabel('Pressure Difference (Pa)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Window trajectory plots saved to: {save_path}")
    plt.show()

def analyze_temporal_patterns(df, save_path='window_analysis_temporal.png'):
    """Analyze temporal patterns within windows."""
    print("\nAnalyzing temporal patterns...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Calculate statistics for each position in the window
    positions = []
    vortex_means = []
    vortex_stds = []
    non_vortex_means = []
    non_vortex_stds = []
    
    for pos in range(60):
        vortex_data = df[(df['label'] == 1) & (df.groupby('window_id').cumcount() == pos)]['PRESSURE']
        non_vortex_data = df[(df['label'] == 0) & (df.groupby('window_id').cumcount() == pos)]['PRESSURE']
        
        positions.append(pos)
        vortex_means.append(vortex_data.mean())
        vortex_stds.append(vortex_data.std())
        non_vortex_means.append(non_vortex_data.mean())
        non_vortex_stds.append(non_vortex_data.std())
    
    # Plot 1: Mean pressure by position
    ax1 = axes[0, 0]
    ax1.plot(positions, vortex_means, 'r-', linewidth=2, label='Vortex')
    ax1.fill_between(positions, 
                     np.array(vortex_means) - np.array(vortex_stds),
                     np.array(vortex_means) + np.array(vortex_stds),
                     alpha=0.3, color='red')
    ax1.plot(positions, non_vortex_means, 'b-', linewidth=2, label='Non-Vortex')
    ax1.fill_between(positions,
                     np.array(non_vortex_means) - np.array(non_vortex_stds),
                     np.array(non_vortex_means) + np.array(non_vortex_stds),
                     alpha=0.3, color='blue')
    ax1.set_title('Mean Pressure by Window Position')
    ax1.set_xlabel('Position in Window (0-59)')
    ax1.set_ylabel('Mean Pressure (Pa)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation by position
    ax2 = axes[0, 1]
    ax2.plot(positions, vortex_stds, 'r-', linewidth=2, label='Vortex')
    ax2.plot(positions, non_vortex_stds, 'b-', linewidth=2, label='Non-Vortex')
    ax2.set_title('Pressure Variability by Window Position')
    ax2.set_xlabel('Position in Window (0-59)')
    ax2.set_ylabel('Standard Deviation (Pa)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Pressure slope analysis
    ax3 = axes[1, 0]
    vortex_slopes = []
    non_vortex_slopes = []
    
    for window_id in df['window_id'].unique():
        window_data = df[df['window_id'] == window_id].sort_values('SCLK')
        if len(window_data) >= 2:
            slope = np.polyfit(range(len(window_data)), window_data['PRESSURE'], 1)[0]
            label = df[df['window_id'] == window_id]['label'].iloc[0]
            if label == 1:
                vortex_slopes.append(slope)
            else:
                non_vortex_slopes.append(slope)
    
    ax3.hist(vortex_slopes, bins=30, alpha=0.7, color='red', label=f'Vortex (n={len(vortex_slopes)})', density=True)
    ax3.hist(non_vortex_slopes, bins=30, alpha=0.7, color='blue', label=f'Non-Vortex (n={len(non_vortex_slopes)})', density=True)
    ax3.set_title('Distribution of Window Slopes')
    ax3.set_xlabel('Pressure Slope (Pa/sample)')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pressure drop analysis
    ax4 = axes[1, 1]
    vortex_drops = []
    non_vortex_drops = []
    
    for window_id in df['window_id'].unique():
        window_data = df[df['window_id'] == window_id].sort_values('SCLK')
        if len(window_data) >= 2:
            pressure_drop = window_data['PRESSURE'].iloc[0] - window_data['PRESSURE'].min()
            label = df[df['window_id'] == window_id]['label'].iloc[0]
            if label == 1:
                vortex_drops.append(pressure_drop)
            else:
                non_vortex_drops.append(pressure_drop)
    
    ax4.hist(vortex_drops, bins=30, alpha=0.7, color='red', label=f'Vortex (n={len(vortex_drops)})', density=True)
    ax4.hist(non_vortex_drops, bins=30, alpha=0.7, color='blue', label=f'Non-Vortex (n={len(non_vortex_drops)})', density=True)
    ax4.set_title('Distribution of Maximum Pressure Drops')
    ax4.set_xlabel('Pressure Drop (Pa)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Temporal pattern analysis saved to: {save_path}")
    plt.show()

def analyze_window_quality(df, save_path='window_analysis_quality.txt'):
    """Analyze window data quality and characteristics."""
    print("\nAnalyzing window quality...")
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("WINDOW-LEVEL DATA QUALITY ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total samples: {len(df):,}\n")
        f.write(f"Unique windows: {df['window_id'].nunique()}\n")
        f.write(f"Expected samples per window: 60\n")
        f.write(f"Actual average samples per window: {len(df) / df['window_id'].nunique():.1f}\n\n")
        
        # Window size analysis
        window_sizes = df.groupby('window_id').size()
        f.write("WINDOW SIZE ANALYSIS:\n")
        f.write(f"  Min window size: {window_sizes.min()}\n")
        f.write(f"  Max window size: {window_sizes.max()}\n")
        f.write(f"  Mean window size: {window_sizes.mean():.1f}\n")
        f.write(f"  Windows with 60 samples: {(window_sizes == 60).sum()}\n")
        f.write(f"  Windows with <60 samples: {(window_sizes < 60).sum()}\n")
        f.write(f"  Windows with >60 samples: {(window_sizes > 60).sum()}\n\n")
        
        # Pressure statistics
        f.write("PRESSURE STATISTICS:\n")
        f.write(f"  Global pressure range: {df['PRESSURE'].min():.3f} to {df['PRESSURE'].max():.3f} Pa\n")
        f.write(f"  Global pressure mean: {df['PRESSURE'].mean():.3f} Pa\n")
        f.write(f"  Global pressure std: {df['PRESSURE'].std():.3f} Pa\n\n")
        
        # Class-specific statistics
        vortex_data = df[df['label'] == 1]
        non_vortex_data = df[df['label'] == 0]
        
        f.write("CLASS-SPECIFIC PRESSURE STATISTICS:\n")
        f.write(f"  Vortex windows:\n")
        f.write(f"    Pressure range: {vortex_data['PRESSURE'].min():.3f} to {vortex_data['PRESSURE'].max():.3f} Pa\n")
        f.write(f"    Pressure mean: {vortex_data['PRESSURE'].mean():.3f} Pa\n")
        f.write(f"    Pressure std: {vortex_data['PRESSURE'].std():.3f} Pa\n")
        f.write(f"  Non-vortex windows:\n")
        f.write(f"    Pressure range: {non_vortex_data['PRESSURE'].min():.3f} to {non_vortex_data['PRESSURE'].max():.3f} Pa\n")
        f.write(f"    Pressure mean: {non_vortex_data['PRESSURE'].mean():.3f} Pa\n")
        f.write(f"    Pressure std: {non_vortex_data['PRESSURE'].std():.3f} Pa\n\n")
        
        # Temporal analysis
        f.write("TEMPORAL ANALYSIS:\n")
        vortex_windows = df[df['label'] == 1]['window_id'].unique()
        non_vortex_windows = df[df['label'] == 0]['window_id'].unique()
        
        vortex_slopes = []
        non_vortex_slopes = []
        
        for window_id in vortex_windows:
            window_data = df[df['window_id'] == window_id].sort_values('SCLK')
            if len(window_data) >= 2:
                slope = np.polyfit(range(len(window_data)), window_data['PRESSURE'], 1)[0]
                vortex_slopes.append(slope)
        
        for window_id in non_vortex_windows:
            window_data = df[df['window_id'] == window_id].sort_values('SCLK')
            if len(window_data) >= 2:
                slope = np.polyfit(range(len(window_data)), window_data['PRESSURE'], 1)[0]
                non_vortex_slopes.append(slope)
        
        f.write(f"  Vortex window slopes: mean={np.mean(vortex_slopes):.6f}, std={np.std(vortex_slopes):.6f}\n")
        f.write(f"  Non-vortex window slopes: mean={np.mean(non_vortex_slopes):.6f}, std={np.std(non_vortex_slopes):.6f}\n")
        f.write(f"  Slope difference: {np.mean(vortex_slopes) - np.mean(non_vortex_slopes):.6f}\n\n")
        
        # Statistical test
        if len(vortex_slopes) > 0 and len(non_vortex_slopes) > 0:
            t_stat, p_value = stats.ttest_ind(vortex_slopes, non_vortex_slopes)
            f.write("STATISTICAL TEST (Slopes):\n")
            f.write(f"  T-statistic: {t_stat:.4f}\n")
            f.write(f"  P-value: {p_value:.6f}\n")
            f.write(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)\n")
    
    print(f"Window quality analysis saved to: {save_path}")

def main():
    """Main analysis function."""
    print("="*70)
    print("COMPREHENSIVE WINDOW-LEVEL ANALYSIS")
    print("="*70)
    print("Analyzing raw 60-sample windows for temporal patterns")
    print("="*70)
    
    # Load data
    df = load_window_data()
    
    # Perform analyses
    window_stats = analyze_window_statistics(df)
    create_window_trajectory_plots(df)
    analyze_temporal_patterns(df)
    analyze_window_quality(df)
    
    print(f"\n{'='*70}")
    print("WINDOW ANALYSIS COMPLETED")
    print(f"{'='*70}")
    print("Generated files:")
    print("  [ANALYSIS] window_analysis_trajectories.png - Pressure trajectories")
    print("  [ANALYSIS] window_analysis_temporal.png - Temporal patterns")
    print("  [ANALYSIS] window_analysis_quality.txt - Data quality report")
    print("\nThis analysis reveals:")
    print("  • How pressure evolves within 60-sample windows")
    print("  • Temporal patterns that distinguish vortex vs non-vortex")
    print("  • Data quality and consistency across windows")
    print("  • Statistical differences in pressure trajectories")
    print("="*70)

if __name__ == "__main__":
    main()












