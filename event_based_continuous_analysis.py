#!/usr/bin/env python3
"""
Event-Based Continuous Validation Analysis
=========================================

This script creates focused visualizations for individual vortex events using
the continuous validation approach. Each plot shows:
- Individual vortex events with focused time windows
- Pressure data (black line)
- Model confidence for precursor prediction (red line)
- All ground truth regions (4xFWHM, detection, FWHM)
- Model predictions and threshold

Optimized with pre-computed features for speed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
VALIDATION_ML_FILE = "temporal_splits/ml_val.csv"
VALIDATION_SLIDING_FEATURES_FILE = "val_sliding_features.csv"
TRAIN_FEATURES_FILE = "train_features.csv"
OUTPUT_DIR = "event_based_continuous_analysis"

# Event window configuration
EVENT_WINDOW_HOURS = 2.0  # ±2 hours around each event
SAMPLES_PER_HOUR = 3600   # Assuming 1Hz sampling

# =============================================================================
# Helper Functions
# =============================================================================

def load_data():
    """Load validation data and train model efficiently."""
    print("Loading data...")
    
    # Load validation ML data
    val_ml = pd.read_csv(VALIDATION_ML_FILE)
    print(f"  Loaded {len(val_ml):,} validation ML samples")
    
    # Load pre-computed sliding features
    val_features = pd.read_csv(VALIDATION_SLIDING_FEATURES_FILE)
    print(f"  Loaded {len(val_features):,} pre-computed sliding window features")
    
    # Train model quickly on training features
    train_features = pd.read_csv(TRAIN_FEATURES_FILE)
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    # Fast model training
    rf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced for speed
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("  Training model (fast version)...")
    rf_model.fit(X_train, y_train)
    print("  Model ready!")
    
    return val_ml, val_features, rf_model, feature_cols

def find_vortex_events(val_ml):
    """Find individual vortex events from ground truth data."""
    print("\nFinding vortex events...")
    
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
                        'time_hours': peak_idx / 3600.0
                    })
                
                in_4xfwhm = False
    
    print(f"  Found {len(events)} complete vortex events")
    return events

def get_continuous_predictions_for_event(val_ml, val_features, model, feature_cols, event):
    """Get continuous predictions for a specific event window."""
    
    # Define time window around the event
    event_center = event['peak_idx']
    window_samples = int(EVENT_WINDOW_HOURS * SAMPLES_PER_HOUR)
    
    start_idx = max(0, event_center - window_samples)
    end_idx = min(len(val_ml), event_center + window_samples)
    
    # Extract event data
    event_data = val_ml.iloc[start_idx:end_idx].copy()
    event_data = event_data.reset_index(drop=True)
    
    # Create time axis in hours
    time_hours = np.arange(len(event_data)) / 3600.0 + (start_idx / 3600.0)
    
    # Get pre-computed features for this time window
    event_features = val_features[
        (val_features['start_idx'] >= start_idx) & 
        (val_features['end_idx'] <= end_idx)
    ].copy()
    
    if len(event_features) == 0:
        print(f"  No pre-computed features found for event {event['event_id']}")
        return None, None, None
    
    # Filter valid predictions
    valid_features = event_features[event_features['label'] != 'Omit'].copy()
    if len(valid_features) == 0:
        print(f"  No valid features found for event {event['event_id']}")
        return None, None, None
    
    # Get predictions
    X_features = valid_features[feature_cols].values
    probabilities = model.predict_proba(X_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    # Convert window positions to time
    window_times = (valid_features['end_idx'] - start_idx) / 3600.0
    
    return event_data, time_hours, window_times, probabilities, predictions

def create_event_focused_plot(event_data, time_hours, window_times, probabilities, predictions, 
                            event, output_path=None):
    """Create a focused visualization for a single vortex event."""
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Left Y-axis: Pressure data
    ax.plot(time_hours, event_data['PRESSURE'].values, 'k-', linewidth=1.5, label='Pressure', alpha=0.8)
    ax.set_xlabel('Time (hours from start)', fontsize=12)
    ax.set_ylabel('Pressure (Pa)', fontsize=12, color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.grid(True, alpha=0.3)
    
    # Right Y-axis: Model confidence
    ax2 = ax.twinx()
    ax2.plot(window_times, probabilities, 'r-', linewidth=2, label='Model Confidence', alpha=0.8)
    ax2.set_ylabel('Model Confidence (0-1)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    # Decision threshold
    threshold = 0.5
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    # Ground truth regions
    event_center = event['peak_idx']
    window_samples = int(EVENT_WINDOW_HOURS * SAMPLES_PER_HOUR)
    start_idx = max(0, event_center - window_samples)
    
    # Convert indices to time
    detection_start_time = (event['detection_start'] - start_idx) / 3600.0
    detection_end_time = (event['detection_end'] - start_idx) / 3600.0
    fwhm_start_time = (event['fwhm_start'] - start_idx) / 3600.0
    fwhm_end_time = (event['fwhm_end'] - start_idx) / 3600.0
    fourxfwhm_start_time = (event['fourxfwhm_start'] - start_idx) / 3600.0
    fourxfwhm_end_time = (event['fourxfwhm_end'] - start_idx) / 3600.0
    peak_time = (event['peak_idx'] - start_idx) / 3600.0
    
    # Only show regions that are within our plot window
    plot_end = len(event_data)/3600.0
    
    # 4xFWHM (Gray - Extended region)
    if 0 <= fourxfwhm_start_time <= plot_end:
        ax.axvspan(fourxfwhm_start_time, fourxfwhm_end_time, alpha=0.15, color='gray', 
                   label='GT 4xFWHM Window')
    
    # Detection Window (Red - Precursor region)
    if 0 <= detection_start_time <= plot_end:
        ax.axvspan(detection_start_time, detection_end_time, alpha=0.25, color='red', 
                   label='GT Detection Window')
    
    # FWHM Window (Green - Core vortex)
    if 0 <= fwhm_start_time <= plot_end:
        ax.axvspan(fwhm_start_time, fwhm_end_time, alpha=0.25, color='green', 
                   label='GT FWHM Window')
    
    # Add vortex event marker (Orange - Peak)
    if 0 <= peak_time <= plot_end:
        ax.axvline(x=peak_time, color='orange', linewidth=3, label='Vortex Event Peak')
    
    # Model predictions as orange scatter points
    pred_mask = predictions == 1
    if pred_mask.any():
        ax2.scatter(window_times[pred_mask], probabilities[pred_mask], 
                   color='orange', s=40, alpha=0.9, zorder=3, 
                   label='Model Predictions', edgecolors='white', linewidth=1)
    
    # Set title with event information
    title = f"Vortex Event {event['event_id']} - Continuous Analysis"
    subtitle = f"Event SCLK: {event['peak_sclk']}, Time: {event['time_hours']:.2f} hours"
    plt.title(f"{title}\n{subtitle}", fontsize=14, pad=20)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved event {event['event_id']} to: {output_path}")
    
    return fig

def analyze_event_performance(event_data, window_times, probabilities, predictions, event):
    """Analyze model performance for a specific event."""
    
    print(f"\nEvent {event['event_id']} Analysis:")
    print(f"  Peak SCLK: {event['peak_sclk']}")
    print(f"  Time window: ±{EVENT_WINDOW_HOURS} hours around peak")
    
    # Model confidence statistics
    if len(probabilities) > 0:
        print(f"  Model confidence - Mean: {probabilities.mean():.3f}, Max: {probabilities.max():.3f}")
        print(f"  Positive predictions: {predictions.sum()}/{len(predictions)} ({predictions.sum()/len(predictions)*100:.1f}%)")
    
    # Ground truth regions analysis
    print(f"  Ground truth regions:")
    print(f"    Detection window: {event['detection_start']} to {event['detection_end']}")
    print(f"    FWHM window: {event['fwhm_start']} to {event['fwhm_end']}")
    print(f"    4xFWHM window: {event['fourxfwhm_start']} to {event['fourxfwhm_end']}")

# =============================================================================
# Main Execution
# =============================================================================

def run_event_based_continuous_analysis(num_events=5):
    """
    Run event-based continuous validation analysis.
    
    Args:
        num_events: Number of events to analyze (default: 5)
    """
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("EVENT-BASED CONTINUOUS VALIDATION ANALYSIS")
    print("="*70)
    
    # 1. Load data and train model
    val_ml, val_features, model, feature_cols = load_data()
    
    # 2. Find vortex events
    events = find_vortex_events(val_ml)
    
    if len(events) == 0:
        print("No vortex events found!")
        return
    
    # 3. Create visualizations for specified number of events
    num_events = min(num_events, len(events))
    print(f"\nCreating event-based visualizations for first {num_events} events...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, event in enumerate(events[:num_events]):
        print(f"\nProcessing Event {i+1}/{num_events}: SCLK {event['peak_sclk']}")
        
        # Get continuous predictions for this event
        result = get_continuous_predictions_for_event(val_ml, val_features, model, feature_cols, event)
        
        if result[0] is not None:
            event_data, time_hours, window_times, probabilities, predictions = result
            
            # Analyze event performance
            analyze_event_performance(event_data, window_times, probabilities, predictions, event)
            
            # Create visualization
            output_file = f"event_{event['event_id']:02d}_continuous_{timestamp}.png"
            output_path = os.path.join(OUTPUT_DIR, output_file)
            
            fig = create_event_focused_plot(
                event_data, time_hours, window_times, probabilities, predictions, 
                event, output_path
            )
            
            plt.close(fig)  # Close to free memory
        else:
            print(f"  Skipping event {event['event_id']} - insufficient data")
    
    print(f"\n{'='*70}")
    print("EVENT-BASED CONTINUOUS ANALYSIS COMPLETED!")
    print(f"{'='*70}")
    print(f"Created {num_events} focused event visualizations")
    print("Each plot shows:")
    print("  - Raw pressure data (black line)")
    print("  - Model confidence for precursor prediction (red line)")
    print("  - Ground truth 4xFWHM window (gray shading)")
    print("  - Ground truth detection window (red shading)")
    print("  - Ground truth FWHM window (green shading)")
    print("  - Vortex event peak marker (orange line)")
    print("  - Model predictions (orange dots)")
    print("  - Decision threshold (red dashed line)")
    print(f"\nPlots saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments for number of events
    num_events = 5  # Default
    
    if len(sys.argv) > 1:
        try:
            num_events = int(sys.argv[1])
            print(f"Analyzing {num_events} events")
        except ValueError:
            print("Usage: python event_based_continuous_analysis.py [num_events]")
            print("num_events should be an integer")
            sys.exit(1)
    
    run_event_based_continuous_analysis(num_events)












