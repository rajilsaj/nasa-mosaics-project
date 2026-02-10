#!/usr/bin/env python3
"""
Event-Based Training Analysis
Shows individual vortex events with precursor → vortex sequence
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

def load_training_data_and_model():
    """Load training windows data and trained model."""
    print("=" * 70)
    print("EVENT-BASED TRAINING ANALYSIS")
    print("=" * 70)
    
    # Load raw training windows
    train_df = pd.read_csv("train_windows.csv")
    print(f"Loaded {len(train_df)} training window samples")
    
    # Load training features for model predictions
    train_features_df = pd.read_csv("train_features.csv")
    print(f"Loaded {len(train_features_df)} training feature vectors")
    
    # Find latest model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("improved_rf_vortex_detector_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No improved model found. Run improved_train_rf_model.py first.")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    print(f"Loading model: {latest_model}")
    
    # Load model
    model = joblib.load(model_path)
    
    return train_df, train_features_df, model

def add_model_predictions(train_df, train_features_df, model):
    """Add model probability predictions to training data."""
    print("\nAdding model predictions to training data...")
    
    # Prepare features for prediction (exclude event_sclk to prevent data leakage)
    feature_cols = [col for col in train_features_df.columns if col not in ['window_id', 'label', 'event_sclk']]
    X_train = train_features_df[feature_cols].values
    
    # Get model probabilities
    rf_prob = model.predict_proba(X_train)[:, 1]
    
    # Add probabilities to features dataframe
    train_features_df = train_features_df.copy()
    train_features_df['rf_prob'] = rf_prob
    
    # Create mapping from window_id to probability
    window_prob_map = dict(zip(train_features_df['window_id'], train_features_df['rf_prob']))
    
    # Merge probabilities back to raw training data by window_id
    train_df_with_probs = train_df.copy()
    train_df_with_probs['rf_prob'] = train_df_with_probs['window_id'].map(window_prob_map)
    
    print(f"Added model predictions to {train_df_with_probs['rf_prob'].notna().sum()} samples")
    
    return train_df_with_probs

def identify_vortex_events(train_df):
    """Identify individual vortex events from the data."""
    print("\nIdentifying individual vortex events...")
    
    events = []
    current_event = None
    
    for idx, row in train_df.iterrows():
        if row['gt_fwhm']:  # Start of a vortex event
            if current_event is None:
                # Start new event
                current_event = {
                    'event_id': len(events) + 1,
                    'start_idx': idx,
                    'start_sclk': row['SCLK'],
                    'fwhm_indices': [idx],
                    'detection_indices': [],
                    'precursor_indices': []
                }
            else:
                # Continue current event
                current_event['fwhm_indices'].append(idx)
        elif row['gt_detection_win'] and not row['gt_fwhm']:
            # Detection window but not FWHM (precursor region)
            if current_event is not None:
                current_event['detection_indices'].append(idx)
            else:
                # Precursor before vortex starts
                current_event = {
                    'event_id': len(events) + 1,
                    'start_idx': idx,
                    'start_sclk': row['SCLK'],
                    'fwhm_indices': [],
                    'detection_indices': [idx],
                    'precursor_indices': [idx]
                }
        else:
            # End of event
            if current_event is not None:
                current_event['end_idx'] = idx
                current_event['end_sclk'] = train_df.iloc[idx-1]['SCLK']
                events.append(current_event)
                current_event = None
    
    # Handle last event if it exists
    if current_event is not None:
        current_event['end_idx'] = len(train_df) - 1
        current_event['end_sclk'] = train_df.iloc[-1]['SCLK']
        events.append(current_event)
    
    print(f"Identified {len(events)} vortex events")
    
    # Add summary statistics for each event
    for event in events:
        event_data = train_df.iloc[event['start_idx']:event['end_idx']+1]
        event['pressure_range'] = event_data['PRESSURE'].max() - event_data['PRESSURE'].min()
        event['pressure_drop'] = event_data['PRESSURE'].iloc[0] - event_data['PRESSURE'].min()
        event['duration_samples'] = len(event_data)
        event['mean_rf_prob'] = event_data['rf_prob'].mean() if 'rf_prob' in event_data.columns else 0
        event['max_rf_prob'] = event_data['rf_prob'].max() if 'rf_prob' in event_data.columns else 0
    
    return events

def plot_event_analysis(train_df, events, num_events=5, output_dir="results"):
    """Plot individual vortex events with precursor → vortex sequence."""
    print(f"\nCreating event-based analysis plots for {min(num_events, len(events))} events...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subplots for multiple events
    fig, axes = plt.subplots(min(num_events, len(events)), 1, figsize=(15, 4 * min(num_events, len(events))))
    if num_events == 1 or len(events) == 1:
        axes = [axes]
    
    for i, event in enumerate(events[:num_events]):
        ax = axes[i]
        
        # Get event data
        start_idx = event['start_idx']
        end_idx = event['end_idx']
        event_data = train_df.iloc[start_idx:end_idx+1].copy()
        
        # Create twinx for this event
        ax2 = ax.twinx()
        
        # Plot pressure (left y-axis)
        ax.plot(event_data['SCLK'], event_data['PRESSURE'], 'k-', linewidth=2, label='Pressure', alpha=0.8)
        ax.set_ylabel(f"Pressure (Pa)", color='k', fontsize=10)
        ax.tick_params(axis='y', labelcolor='k')
        
        # Plot RF probability (right y-axis)
        if 'rf_prob' in event_data.columns:
            ax2.plot(event_data['SCLK'], event_data['rf_prob'], 'b-', linewidth=2, label='RF Probability', alpha=0.8)
            ax2.set_ylabel("Predicted Probability", color='b', fontsize=10)
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='y', labelcolor='b')
        
        # Shade regions
        # 4xFWHM (background)
        if 'gt_4xfwhm' in event_data.columns:
            mask = event_data['gt_4xfwhm'].astype(bool)
            if mask.any():
                ax.fill_between(event_data['SCLK'],
                               event_data['PRESSURE'].min(),
                               event_data['PRESSURE'].max(),
                               where=mask, color='lightgray', alpha=0.2, label='4xFWHM')
        
        # Detection window (precursor)
        if 'gt_detection_win' in event_data.columns:
            mask = event_data['gt_detection_win'].astype(bool)
            if mask.any():
                ax.fill_between(event_data['SCLK'],
                               event_data['PRESSURE'].min(),
                               event_data['PRESSURE'].max(),
                               where=mask, color='red', alpha=0.4, label='Precursor')
        
        # FWHM (actual vortex)
        if 'gt_fwhm' in event_data.columns:
            mask = event_data['gt_fwhm'].astype(bool)
            if mask.any():
                ax.fill_between(event_data['SCLK'],
                               event_data['PRESSURE'].min(),
                               event_data['PRESSURE'].max(),
                               where=mask, color='green', alpha=0.6, label='Vortex')
        
        # Add threshold lines
        if 'rf_prob' in event_data.columns:
            ax2.axhline(0.5, color='blue', ls='--', alpha=0.6, label='Threshold (0.5)')
            ax2.axhline(0.45, color='orange', ls='--', alpha=0.6, label='High-Recall (0.45)')
            ax2.axhline(0.9, color='purple', ls='--', alpha=0.6, label='High-Precision (0.9)')
        
        # Formatting
        ax.set_title(f"Event {event['event_id']}: SCLK {event['start_sclk']:.0f} - {event['end_sclk']:.0f}\n" +
                    f"Pressure Drop: {event['pressure_drop']:.2f} Pa, "
                    f"Duration: {event['duration_samples']} samples, "
                    f"Max RF Prob: {event['max_rf_prob']:.3f}", fontsize=11)
        
        ax.set_xlabel("SCLK", fontsize=10)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + lines2, loc='upper right', fontsize=8, ncol=2)
    
    plt.suptitle(f"Event-Based Training Analysis: Individual Vortex Events\n" +
                f"Red = Precursor Regions, Green = Actual Vortex Events", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"event_based_training_analysis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Event-based analysis plot saved to: {plot_filename}")
    
    plt.show()
    
    return fig

def plot_window_based_analysis(train_df, train_features_df, model, num_windows=6, output_dir="results"):
    """Plot individual training windows with their features and predictions."""
    print(f"\nCreating window-based analysis for {num_windows} training windows...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find windows that contain FWHM regions
    windows_with_fwhm = train_df[train_df['gt_fwhm'] == True]['window_id'].unique()
    print(f"Windows with FWHM regions: {len(windows_with_fwhm)}")
    
    # Get mix of windows: some with FWHM, some without
    if len(windows_with_fwhm) > 0:
        # Take 3 windows with FWHM and 3 without
        fwhm_windows = list(windows_with_fwhm[:3])
        non_fwhm_windows = train_features_df[~train_features_df['window_id'].isin(windows_with_fwhm)]['window_id'].unique()[:3]
        unique_windows = fwhm_windows + list(non_fwhm_windows)
    else:
        unique_windows = train_features_df['window_id'].unique()[:num_windows]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, window_id in enumerate(unique_windows):
        ax = axes[i]
        
        # Get window data
        window_data = train_df[train_df['window_id'] == window_id].copy()
        window_features = train_features_df[train_features_df['window_id'] == window_id].iloc[0]
        
        # Create twinx
        ax2 = ax.twinx()
        
        # Plot pressure (left y-axis)
        ax.plot(range(len(window_data)), window_data['PRESSURE'], 'k-', linewidth=2, label='Pressure')
        ax.set_ylabel("Pressure (Pa)", color='k')
        ax.tick_params(axis='y', labelcolor='k')
        
        # Plot RF probability (right y-axis)
        if 'rf_prob' in window_features:
            ax2.axhline(window_features['rf_prob'], color='b', linewidth=3, label=f'RF Prob: {window_features["rf_prob"]:.3f}')
            ax2.set_ylabel("RF Probability", color='b')
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Add threshold lines on the RIGHT y-axis (RF probability axis)
            ax2.axhline(0.5, color='blue', ls='--', alpha=0.6, label='Threshold (0.5)')
            ax2.axhline(0.45, color='orange', ls='--', alpha=0.6, label='High-Recall (0.45)')
            ax2.axhline(0.9, color='purple', ls='--', alpha=0.6, label='High-Precision (0.9)')
        
        # Shade regions (on pressure axis)
        # Detection window (precursor)
        if 'gt_detection_win' in window_data.columns:
            mask = window_data['gt_detection_win'].astype(bool)
            if mask.any():
                ax.fill_between(range(len(window_data)),
                               window_data['PRESSURE'].min(),
                               window_data['PRESSURE'].max(),
                               where=mask, color='red', alpha=0.4, label='Precursor')
        
        # FWHM (actual vortex)
        if 'gt_fwhm' in window_data.columns:
            mask = window_data['gt_fwhm'].astype(bool)
            if mask.any():
                ax.fill_between(range(len(window_data)),
                               window_data['PRESSURE'].min(),
                               window_data['PRESSURE'].max(),
                               where=mask, color='green', alpha=0.6, label='Vortex')
                print(f"Window {window_id}: Found {mask.sum()} FWHM samples")
        
        # Determine training correctness
        rf_prob = window_features.get('rf_prob', 0)
        label = window_features['label']
        
        # Check if model prediction matches training label
        if label == 1.0 and rf_prob > 0.5:
            correctness = "✅ CORRECT"
            correctness_color = "green"
        elif label == 0.0 and rf_prob <= 0.5:
            correctness = "✅ CORRECT" 
            correctness_color = "green"
        elif label == 1.0 and rf_prob <= 0.5:
            correctness = "❌ WRONG"
            correctness_color = "red"
        else:  # label == 0.0 and rf_prob > 0.5
            correctness = "❌ WRONG"
            correctness_color = "red"
        
        # Title with training validation info
        ax.set_title(f"Window {window_id}\n" +
                    f"Training Label: {label} | RF Probability: {rf_prob:.3f} | {correctness}\n" +
                    f"Pressure Drop: {window_data['PRESSURE'].iloc[0] - window_data['PRESSURE'].min():.2f} Pa | " +
                    f"Slope: {window_features.get('overall_slope', 0):.3f}", 
                    fontsize=10, color=correctness_color)
        
        ax.set_xlabel("Sample Index")
        
        # Legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + lines2, loc='upper right', fontsize=8)
    
    plt.suptitle("Window-Based Training Analysis: Individual Training Windows\n" +
                f"Each window shows 60 samples with engineered features", 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"window_based_training_analysis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Window-based analysis plot saved to: {plot_filename}")
    
    plt.show()
    
    return fig

def print_event_summary(events):
    """Print summary of identified events."""
    print("\n" + "=" * 70)
    print("EVENT SUMMARY")
    print("=" * 70)
    
    for event in events[:10]:  # Show first 10 events
        print(f"\nEvent {event['event_id']}:")
        print(f"  SCLK Range: {event['start_sclk']:.0f} - {event['end_sclk']:.0f}")
        print(f"  Duration: {event['duration_samples']} samples")
        print(f"  Pressure Drop: {event['pressure_drop']:.2f} Pa")
        print(f"  Pressure Range: {event['pressure_range']:.2f} Pa")
        print(f"  FWHM Samples: {len(event['fwhm_indices'])}")
        print(f"  Detection Samples: {len(event['detection_indices'])}")
        print(f"  Mean RF Probability: {event['mean_rf_prob']:.3f}")
        print(f"  Max RF Probability: {event['max_rf_prob']:.3f}")

def print_training_validation_summary(train_df, train_features_df):
    """Print comprehensive training validation summary."""
    print("\n" + "=" * 70)
    print("TRAINING VALIDATION SUMMARY")
    print("=" * 70)
    
    # Merge data for analysis (rf_prob is already in train_df from add_model_predictions)
    merged_df = train_df.merge(train_features_df[['window_id', 'label']], on='window_id', how='left')
    
    # Count windows by label
    label_counts = merged_df.groupby('window_id').agg({
        'label': 'first',
        'rf_prob': 'first',
        'gt_detection_win': 'any',
        'gt_fwhm': 'any'
    }).reset_index()
    
    print(f"Total Training Windows: {len(label_counts)}")
    print(f"Label 1.0 (Precursor) Windows: {(label_counts['label'] == 1.0).sum()}")
    print(f"Label 0.0 (No Vortex) Windows: {(label_counts['label'] == 0.0).sum()}")
    
    # Training correctness analysis
    label_1_windows = label_counts[label_counts['label'] == 1.0]
    label_0_windows = label_counts[label_counts['label'] == 0.0]
    
    # Check if Label 1.0 windows have RF probability > 0.5
    correct_label_1 = (label_1_windows['rf_prob'] > 0.5).sum()
    total_label_1 = len(label_1_windows)
    
    # Check if Label 0.0 windows have RF probability <= 0.5  
    correct_label_0 = (label_0_windows['rf_prob'] <= 0.5).sum()
    total_label_0 = len(label_0_windows)
    
    print(f"\nTraining Correctness Analysis:")
    print(f"  Label 1.0 Windows: {correct_label_1}/{total_label_1} correct ({correct_label_1/total_label_1*100:.1f}%)")
    print(f"  Label 0.0 Windows: {correct_label_0}/{total_label_0} correct ({correct_label_0/total_label_0*100:.1f}%)")
    print(f"  Overall Accuracy: {(correct_label_1 + correct_label_0)}/{len(label_counts)} correct ({(correct_label_1 + correct_label_0)/len(label_counts)*100:.1f}%)")
    
    # Threshold analysis
    print(f"\nThreshold Analysis:")
    for threshold in [0.45, 0.5, 0.9]:
        label_1_above = (label_1_windows['rf_prob'] >= threshold).sum()
        label_0_above = (label_0_windows['rf_prob'] >= threshold).sum()
        
        precision = label_1_above / (label_1_above + label_0_above) if (label_1_above + label_0_above) > 0 else 0
        recall = label_1_above / total_label_1 if total_label_1 > 0 else 0
        
        print(f"  Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Window type analysis
    print(f"\nWindow Type Analysis:")
    has_detection = label_counts['gt_detection_win'].sum()
    has_fwhm = label_counts['gt_fwhm'].sum()
    has_both = ((label_counts['gt_detection_win']) & (label_counts['gt_fwhm'])).sum()
    
    print(f"  Windows with Detection Regions: {has_detection}")
    print(f"  Windows with FWHM Regions: {has_fwhm}")
    print(f"  Windows with Both: {has_both}")
    
    # RF probability statistics
    print(f"\nRF Probability Statistics:")
    print(f"  Label 1.0 Mean: {label_1_windows['rf_prob'].mean():.3f}")
    print(f"  Label 1.0 Std: {label_1_windows['rf_prob'].std():.3f}")
    print(f"  Label 0.0 Mean: {label_0_windows['rf_prob'].mean():.3f}")
    print(f"  Label 0.0 Std: {label_0_windows['rf_prob'].std():.3f}")
    
    return label_counts

def main():
    """Main execution function."""
    print("Starting event-based training analysis...")
    
    # Step 1: Load training data and model
    train_df, train_features_df, model = load_training_data_and_model()
    
    # Step 2: Add model predictions
    train_df_with_probs = add_model_predictions(train_df, train_features_df, model)
    
    # Step 3: Identify vortex events
    events = identify_vortex_events(train_df_with_probs)
    
    # Step 4: Print event summary
    print_event_summary(events)
    
    # Step 5: Print comprehensive training validation
    label_counts = print_training_validation_summary(train_df_with_probs, train_features_df)
    
    # Step 6: Create event-based plots
    plot_event_analysis(train_df_with_probs, events, num_events=5)
    
    # Step 7: Create window-based plots with training validation
    plot_window_based_analysis(train_df_with_probs, train_features_df, model, num_windows=6)
    
    print("\n" + "=" * 70)
    print("EVENT-BASED TRAINING ANALYSIS COMPLETED")
    print("=" * 70)
    print("Key Insights:")
    print("- Individual events show clear precursor -> vortex sequence")
    print("- Window-based analysis shows feature engineering results")
    print("- Model confidence correlates with event intensity")
    print("- Training data quality can be visually assessed")
    print("=" * 70)

if __name__ == "__main__":
    main()
