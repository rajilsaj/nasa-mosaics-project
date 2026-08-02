#!/usr/bin/env python3
"""
Fast Event-Based Vortex Visualization
====================================

This script creates focused visualizations for individual vortex events
using pre-computed features to avoid the slow training process.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_precomputed_data():
    """Load pre-computed sliding window features and saved model."""
    print("Loading pre-computed data...")
    
    # Load pre-computed validation sliding features
    val_features = pd.read_csv('datasets/val_sliding_features.csv')
    print(f"  Loaded {len(val_features):,} pre-computed sliding window features")
    
    # Load validation ML data for ground truth
    val_ml = pd.read_csv('datasets/temporal_splits/ml_val.csv')
    print(f"  Loaded {len(val_ml):,} validation ML samples")
    
    # Try to load saved model first (OPTIMIZATION: Don't re-train!)
    import joblib
    model_path = "models/rf_vortex_detector_saved.pkl"
    
    if os.path.exists(model_path):
        print("  Loading saved model...")
        rf_model = joblib.load(model_path)
        print("  Model loaded from cache!")
    else:
        print("  Training model once and saving...")
        train_features = pd.read_csv('datasets/train_features.csv')
        feature_cols = [col for col in train_features.columns 
                       if col not in ['window_id', 'event_sclk', 'label']]
        X_train = train_features[feature_cols].values
        y_train = train_features['label'].values
        
        rf_model = RandomForestClassifier(
            n_estimators=100,  # Full model for production
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Save model for future use (OPTIMIZATION!)
        os.makedirs('models', exist_ok=True)
        joblib.dump(rf_model, model_path)
        print(f"  Model saved to {model_path}")
    
    # Get feature columns
    feature_cols = [col for col in val_features.columns 
                   if col not in ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label']]
    
    return val_features, val_ml, rf_model, feature_cols

def find_vortex_events_fast(val_ml, val_features):
    """Find vortex events using pre-computed features."""
    print("\nFinding vortex events...")
    
    events = []
    
    # Find 4xFWHM regions
    in_4xfwhm = False
    fourxfwhm_start = None
    
    for i, row in val_ml.iterrows():
        if row['gt_4xfwhm'] == True:
            if not in_4xfwhm:
                fourxfwhm_start = i
                in_4xfwhm = True
        else:
            if in_4xfwhm:
                fourxfwhm_end = i - 1
                
                # Find nested regions
                fwhm_start = None
                fwhm_end = None
                detection_start = None
                detection_end = None
                
                # Find FWHM region
                for j in range(fourxfwhm_start, fourxfwhm_end + 1):
                    if val_ml.iloc[j]['gt_fwhm'] == True:
                        if fwhm_start is None:
                            fwhm_start = j
                        fwhm_end = j
                
                # Find detection window
                if fwhm_start is not None:
                    for j in range(fwhm_start - 1, max(fourxfwhm_start, fwhm_start - 200), -1):
                        if val_ml.iloc[j]['gt_detection_win'] == True:
                            detection_end = j
                            break
                    
                    if detection_end is not None:
                        for j in range(detection_end, max(fourxfwhm_start, detection_end - 200), -1):
                            if val_ml.iloc[j]['gt_detection_win'] == True:
                                detection_start = j
                            else:
                                break
                
                if fwhm_start is not None and fwhm_end is not None:
                    # Find peak
                    fwhm_data = val_ml.iloc[fwhm_start:fwhm_end+1]
                    peak_idx = fwhm_data['PRESSURE'].idxmin()
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
    
    print(f"  Found {len(events)} vortex events")
    return events

def create_fast_event_plot(val_ml, val_features, rf_model, feature_cols, event):
    """Create focused visualization using pre-computed features."""
    
    # Define time window around event (±1 hour for speed)
    event_center = event['peak_idx']
    window_hours = 1.0  # ±1 hour
    window_samples = int(window_hours * 3600)
    
    start_idx = max(0, event_center - window_samples)
    end_idx = min(len(val_ml), event_center + window_samples)
    
    # Extract event data
    event_data = val_ml.iloc[start_idx:end_idx].copy()
    event_data = event_data.reset_index(drop=True)
    time_hours = np.arange(len(event_data)) / 3600.0 + (start_idx / 3600.0)
    
    # Get pre-computed features for this time window
    # Find features that overlap with our time window
    event_features = val_features[
        (val_features['start_idx'] >= start_idx) & 
        (val_features['end_idx'] <= end_idx)
    ].copy()
    
    if len(event_features) == 0:
        print(f"  No pre-computed features found for event {event['event_id']}")
        return None
    
    # Sort by position
    event_features = event_features.sort_values('start_idx')
    
    # Get probabilities
    X_features = event_features[feature_cols].values
    probabilities = rf_model.predict_proba(X_features)[:, 1]
    
    # Convert window positions to time
    window_times = (event_features['end_idx'] - start_idx) / 3600.0
    
    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Pressure data
    ax1.plot(time_hours, event_data['PRESSURE'].values, 'b-', linewidth=1.5, label='Pressure', alpha=0.8)
    ax1.set_xlabel('Time (hours from start)', fontsize=12)
    ax1.set_ylabel('Pressure (Pa)', fontsize=12, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    # Model confidence
    ax2 = ax1.twinx()
    ax2.plot(window_times, probabilities, 'r-', linewidth=2, label='Model Confidence', alpha=0.8)
    ax2.set_ylabel('Model Confidence', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1)
    
    # Decision threshold
    threshold = 0.5
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    
    # Ground truth regions
    detection_start_time = (event['detection_start'] - start_idx) / 3600.0
    detection_end_time = (event['detection_end'] - start_idx) / 3600.0
    fwhm_start_time = (event['fwhm_start'] - start_idx) / 3600.0
    fwhm_end_time = (event['fwhm_end'] - start_idx) / 3600.0
    fourxfwhm_start_time = (event['fourxfwhm_start'] - start_idx) / 3600.0
    fourxfwhm_end_time = (event['fourxfwhm_end'] - start_idx) / 3600.0
    peak_time = (event['peak_idx'] - start_idx) / 3600.0
    
    # Add regions (only if within plot window)
    plot_end = len(event_data)/3600.0
    
    if 0 <= fourxfwhm_start_time <= plot_end:
        ax1.axvspan(fourxfwhm_start_time, fourxfwhm_end_time, alpha=0.15, color='gray', 
                   label='GT 4xFWHM Window')
    
    if 0 <= detection_start_time <= plot_end:
        ax1.axvspan(detection_start_time, detection_end_time, alpha=0.25, color='red', 
                   label='GT Detection Window')
    
    if 0 <= fwhm_start_time <= plot_end:
        ax1.axvspan(fwhm_start_time, fwhm_end_time, alpha=0.25, color='green', 
                   label='GT FWHM Window')
    
    if 0 <= peak_time <= plot_end:
        ax1.axvline(x=peak_time, color='orange', linewidth=3, label='Vortex Event Peak')
    
    # Title
    title = f"Vortex Event {event['event_id']} - Fast Analysis"
    subtitle = f"Event SCLK: {event['peak_sclk']}, Time: {event['time_hours']:.2f} hours"
    plt.title(f"{title}\n{subtitle}", fontsize=14, pad=20)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_file = f"fast_event_{event['event_id']:02d}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved event {event['event_id']} to: {output_file}")
    
    return fig

def main():
    """Main execution function."""
    print("="*70)
    print("FAST EVENT-BASED VORTEX VISUALIZATION")
    print("="*70)
    
    try:
        # Load pre-computed data
        val_features, val_ml, rf_model, feature_cols = load_precomputed_data()
        
        # Find events
        events = find_vortex_events_fast(val_ml, val_features)
        
        # Create visualizations for first 3 events
        num_events = min(3, len(events))
        print(f"\nCreating fast visualizations for first {num_events} events...")
        
        for i, event in enumerate(events[:num_events]):
            print(f"\nProcessing Event {i+1}/{num_events}: SCLK {event['peak_sclk']}")
            fig = create_fast_event_plot(val_ml, val_features, rf_model, feature_cols, event)
            if fig:
                plt.close(fig)
        
        print(f"\n{'='*70}")
        print("FAST EVENT-BASED ANALYSIS COMPLETED!")
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
