#!/usr/bin/env python3
"""
Event-Based Vortex Visualization
================================

This script creates focused visualizations for individual vortex events,
similar to the second screenshot. Each plot shows:
- Raw pressure data for a single event
- Model confidence/probability 
- Ground truth windows (detection and FWHM)
- Event timing and SCLK information

This makes it much easier to interpret the model's behavior on specific events.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Feature engineering functions (same as before)
def compute_trend_features(pressure_values):
    """Compute trend features for pressure decrease detection."""
    features = {}
    
    # Overall slope (primary signal)
    x = np.arange(len(pressure_values))
    slope, _ = np.polyfit(x, pressure_values, 1)
    features['overall_slope'] = slope
    
    # First half slope
    h = len(pressure_values) // 2
    x_first = np.arange(h)
    slope_first, _ = np.polyfit(x_first, pressure_values[:h], 1)
    features['first_half_slope'] = slope_first
    
    # Second half slope
    x_second = np.arange(h)
    slope_second, _ = np.polyfit(x_second, pressure_values[h:], 1)
    features['second_half_slope'] = slope_second
    
    # Trend consistency (difference between halves)
    features['trend_consistency'] = abs(slope_first - slope_second)
    
    return features

def compute_pressure_drop_features(pressure_values):
    """Compute pressure drop magnitude and characteristics."""
    features = {}
    
    # Pressure drop (start to minimum)
    pressure_drop = pressure_values[0] - np.min(pressure_values)
    features['pressure_drop'] = pressure_drop
    
    # Drop rate (pressure drop per sample)
    min_idx = np.argmin(pressure_values)
    if min_idx > 0:
        features['drop_rate'] = pressure_drop / min_idx
    else:
        features['drop_rate'] = 0.0
    
    # Position of minimum (normalized)
    features['min_position'] = min_idx / len(pressure_values)
    
    return features

def compute_core_statistics(pressure_values):
    """Compute basic statistical features."""
    features = {}
    
    features['mean'] = np.mean(pressure_values)
    features['std'] = np.std(pressure_values)
    features['range'] = np.max(pressure_values) - np.min(pressure_values)
    
    return features

def compute_temporal_evolution_features(pressure_values):
    """Compute temporal evolution features."""
    features = {}
    
    h = len(pressure_values) // 2
    
    # First and second half means
    features['first_half_mean'] = np.mean(pressure_values[:h])
    features['second_half_mean'] = np.mean(pressure_values[h:])
    
    # Mean ratio (second/first)
    if features['first_half_mean'] != 0:
        features['mean_ratio'] = features['second_half_mean'] / features['first_half_mean']
    else:
        features['mean_ratio'] = 1.0
    
    return features

def compute_anomaly_features(pressure_values, global_mean, global_std):
    """Compute anomaly detection features."""
    features = {}
    
    # Minimum z-score (how unusual is the lowest pressure?)
    min_pressure = np.min(pressure_values)
    features['min_zscore'] = (min_pressure - global_mean) / global_std
    
    # Anomaly strength (how much does the minimum deviate?)
    features['anomaly_strength'] = abs(features['min_zscore'])
    
    return features

def engineer_features_for_window(window_data, global_mean, global_std):
    """Engineer features for a single window (same as training)."""
    if len(window_data) < 60:
        return None
    
    pressure = window_data['PRESSURE'].values
    
    features = {}
    
    # Trend features
    trend_feats = compute_trend_features(pressure)
    features.update(trend_feats)
    
    # Pressure drop features
    drop_feats = compute_pressure_drop_features(pressure)
    features.update(drop_feats)
    
    # Core statistics
    stats_feats = compute_core_statistics(pressure)
    features.update(stats_feats)
    
    # Temporal evolution
    temporal_feats = compute_temporal_evolution_features(pressure)
    features.update(temporal_feats)
    
    # Anomaly detection
    anomaly_feats = compute_anomaly_features(pressure, global_mean, global_std)
    features.update(anomaly_feats)
    
    return features

def load_model_and_data():
    """Load the trained RF model and validation data."""
    print("Loading model and data...")
    
    # Train model directly (avoid pickle issues)
    from sklearn.ensemble import RandomForestClassifier
    
    # Load training features
    train_features = pd.read_csv('datasets/train_features.csv')
    print(f"  Loaded training features: {len(train_features):,} samples")
    
    # Prepare training data
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    print(f"  Training features: {len(feature_cols)}")
    print(f"  Training samples: {len(X_train)}")
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    print("  Model training completed!")
    
    # Load validation ML data
    val_ml = pd.read_csv('datasets/temporal_splits/ml_val.csv')
    print(f"  Loaded validation data: {len(val_ml):,} samples")
    
    # Get global statistics
    global_mean = train_features['mean'].mean()
    global_std = train_features['std'].mean()
    
    print(f"  Global mean: {global_mean:.4f}")
    print(f"  Global std: {global_std:.4f}")
    
    return rf_model, val_ml, global_mean, global_std, feature_cols

def find_vortex_events(val_ml):
    """Find individual vortex events from ground truth data with all regions."""
    print("\nFinding vortex events with all ground truth regions...")
    
    events = []
    
    # Find 4xFWHM regions (extended vortex events)
    in_4xfwhm = False
    fourxfwhm_start = None
    
    for i, row in val_ml.iterrows():
        if row['gt_4xfwhm'] == True:
            if not in_4xfwhm:
                fourxfwhm_start = i
                in_4xfwhm = True
        else:
            if in_4xfwhm:
                # End of 4xFWHM region
                fourxfwhm_end = i - 1
                
                # Find nested regions within this 4xFWHM
                fwhm_start = None
                fwhm_end = None
                detection_start = None
                detection_end = None
                
                # Find FWHM region within 4xFWHM
                for j in range(fourxfwhm_start, fourxfwhm_end + 1):
                    if val_ml.iloc[j]['gt_fwhm'] == True:
                        if fwhm_start is None:
                            fwhm_start = j
                        fwhm_end = j
                
                # Find detection window (should be before FWHM)
                if fwhm_start is not None:
                    # Look backwards from FWHM start for detection window
                    for j in range(fwhm_start - 1, max(fourxfwhm_start, fwhm_start - 200), -1):
                        if val_ml.iloc[j]['gt_detection_win'] == True:
                            detection_end = j
                            break
                    
                    # Find detection start
                    if detection_end is not None:
                        for j in range(detection_end, max(fourxfwhm_start, detection_end - 200), -1):
                            if val_ml.iloc[j]['gt_detection_win'] == True:
                                detection_start = j
                            else:
                                break
                
                if fwhm_start is not None and fwhm_end is not None:
                    # Find the peak SCLK (center of FWHM)
                    fwhm_data = val_ml.iloc[fwhm_start:fwhm_end+1]
                    peak_idx = fwhm_data['PRESSURE'].idxmin()  # Lowest pressure = peak
                    peak_sclk = val_ml.iloc[peak_idx]['SCLK']
                    
                    events.append({
                        'event_id': len(events) + 1,
                        'peak_sclk': peak_sclk,
                        'peak_idx': peak_idx,
                        'detection_start': detection_start,
                        'detection_end': detection_end,
                        'fwhm_start': fwhm_start,
                        'fwhm_end': fwhm_end,
                        'fourxfwhm_start': fourxfwhm_start,
                        'fourxfwhm_end': fourxfwhm_end,
                        'time_hours': peak_idx / 3600.0  # Convert to hours (assuming 1Hz sampling)
                    })
                
                in_4xfwhm = False
    
    print(f"  Found {len(events)} complete vortex events")
    return events

def create_event_visualization(val_ml, rf_model, feature_cols, global_mean, global_std, event, window_size=60, step_size=1):
    """Create a focused visualization for a single vortex event."""
    
    # Define the time window around the event (±2 hours)
    event_center = event['peak_idx']
    window_hours = 2.0  # ±2 hours
    window_samples = int(window_hours * 3600)  # Convert to samples (assuming 1Hz)
    
    start_idx = max(0, event_center - window_samples)
    end_idx = min(len(val_ml), event_center + window_samples)
    
    # Extract the data for this event
    event_data = val_ml.iloc[start_idx:end_idx].copy()
    event_data = event_data.reset_index(drop=True)
    
    # Create time axis in hours
    time_hours = np.arange(len(event_data)) / 3600.0 + (start_idx / 3600.0)
    
    # Create sliding windows for this event with fine step size
    probabilities = []
    window_positions = []
    
    for i in range(0, len(event_data) - window_size + 1, step_size):
        # Extract window
        window_data = event_data.iloc[i:i + window_size].copy()
        
        # Engineer features
        features = engineer_features_for_window(window_data, global_mean, global_std)
        
        if features is not None:
            # Prepare feature vector
            feature_vector = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            
            # Get probability
            prob = rf_model.predict_proba(feature_vector)[0, 1]
            
            probabilities.append(prob)
            window_positions.append(time_hours[i + window_size - 1])  # Right edge of window
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Left Y-axis: Pressure data
    ax1.plot(time_hours, event_data['PRESSURE'].values, 'b-', linewidth=1.5, label='Pressure', alpha=0.8)
    ax1.set_xlabel('Time (hours from start)', fontsize=12)
    ax1.set_ylabel('Pressure (Pa)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Right Y-axis: Model confidence
    ax2 = ax1.twinx()
    ax2.plot(window_positions, probabilities, 'r-', linewidth=2, label='Model Confidence', alpha=0.8)
    ax2.set_ylabel('Model Confidence', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    # Add decision threshold
    threshold = 0.5
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    # Add ground truth regions
    # Convert indices to time
    detection_start_time = (event['detection_start'] - start_idx) / 3600.0
    detection_end_time = (event['detection_end'] - start_idx) / 3600.0
    fwhm_start_time = (event['fwhm_start'] - start_idx) / 3600.0
    fwhm_end_time = (event['fwhm_end'] - start_idx) / 3600.0
    fourxfwhm_start_time = (event['fourxfwhm_start'] - start_idx) / 3600.0
    fourxfwhm_end_time = (event['fourxfwhm_end'] - start_idx) / 3600.0
    peak_time = (event['peak_idx'] - start_idx) / 3600.0
    
    # Only show regions that are within our plot window
    # 4xFWHM (Gray - Extended region)
    if 0 <= fourxfwhm_start_time <= len(event_data)/3600.0:
        ax1.axvspan(fourxfwhm_start_time, fourxfwhm_end_time, alpha=0.15, color='gray', 
                   label='GT 4xFWHM Window')
    
    # Detection Window (Red - Precursor region)
    if 0 <= detection_start_time <= len(event_data)/3600.0:
        ax1.axvspan(detection_start_time, detection_end_time, alpha=0.25, color='red', 
                   label='GT Detection Window')
    
    # FWHM Window (Green - Core vortex)
    if 0 <= fwhm_start_time <= len(event_data)/3600.0:
        ax1.axvspan(fwhm_start_time, fwhm_end_time, alpha=0.25, color='green', 
                   label='GT FWHM Window')
    
    # Add vortex event marker (Orange - Peak)
    if 0 <= peak_time <= len(event_data)/3600.0:
        ax1.axvline(x=peak_time, color='orange', linewidth=3, label='Vortex Event Peak')
    
    # Set title with event information
    title = f"FWHM Event {event['event_id']} - Focused Analysis"
    subtitle = f"Event SCLK: {event['peak_sclk']}, Time: {event['time_hours']:.2f} hours"
    plt.title(f"{title}\n{subtitle}", fontsize=14, pad=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = f"event_{event['event_id']:02d}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved event {event['event_id']} visualization to: {output_file}")
    
    return fig, probabilities, window_positions

def main():
    """Main execution function."""
    print("="*70)
    print("EVENT-BASED VORTEX VISUALIZATION")
    print("="*70)
    
    try:
        # Load model and data
        rf_model, val_ml, global_mean, global_std, feature_cols = load_model_and_data()
        
        # Find vortex events
        events = find_vortex_events(val_ml)
        
        # Create visualizations for first 5 events
        num_events = min(5, len(events))
        print(f"\nCreating focused visualizations for first {num_events} events...")
        
        for i, event in enumerate(events[:num_events]):
            print(f"\nProcessing Event {i+1}/{num_events}: SCLK {event['peak_sclk']}")
            fig, probs, positions = create_event_visualization(
                val_ml, rf_model, feature_cols, global_mean, global_std, event)
            plt.close(fig)  # Close to free memory
        
        print(f"\n{'='*70}")
        print("EVENT-BASED ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Created {num_events} focused event visualizations")
        print("Each plot shows:")
        print("  - Raw pressure data (blue line)")
        print("  - Model confidence for precursor prediction (red line)")
        print("  - Ground truth 4xFWHM window (gray shading)")
        print("  - Ground truth detection window (red shading)")
        print("  - Ground truth FWHM window (green shading)")
        print("  - Vortex event peak marker (orange line)")
        print("  - Decision threshold (red dashed line)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
