#!/usr/bin/env python3
"""
Event-Based Validation Analysis
Shows individual validation windows with pressure vs Random Forest probability
"""

import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# DATA LOADING
# =============================================================================

def load_validation_data_and_model():
    """Load validation raw windows, engineered features, and trained model."""
    print("=" * 70)
    print("EVENT-BASED VALIDATION ANALYSIS")
    print("=" * 70)

    # Load raw validation windows (60-sample precursor windows)
    val_windows = pd.read_csv("datasets/val_windows.csv")
    print(f"Loaded {len(val_windows):,} validation window samples")
    print(f"Unique windows: {val_windows['window_id'].nunique()}")

    # Load validation engineered features
    val_features = pd.read_csv("datasets/val_features.csv")
    print(f"Loaded {len(val_features):,} validation feature vectors")

    # Find latest improved model
    models_dir = "models"
    model_files = [
        f for f in os.listdir(models_dir)
        if f.startswith("improved_rf_vortex_detector_") and f.endswith(".pkl")
    ]
    if not model_files:
        raise FileNotFoundError("No improved model found. Run improved_train_rf_model.py first.")

    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    print(f"Loading model: {latest_model}")
    model = joblib.load(model_path)

    return val_windows, val_features, model

# =============================================================================
# MODEL PREDICTIONS
# =============================================================================

def add_model_predictions(val_windows, val_features, model):
    """Add Random Forest probabilities to validation windows."""
    print("\nAdding model predictions to validation data...")

    # Prepare feature matrix matching training order
    train_feature_cols = [
        col for col in pd.read_csv("datasets/train_features.csv").columns
        if col not in ["window_id", "label", "event_sclk"]
    ]
    feature_cols = [col for col in train_feature_cols if col in val_features.columns]

    X_val = val_features[feature_cols].values
    rf_prob = model.predict_proba(X_val)[:, 1]

    # Attach probabilities to features dataframe
    val_features = val_features.copy()
    val_features['rf_prob'] = rf_prob

    # Map probability and label back to raw window samples
    prob_map = dict(zip(val_features['window_id'], val_features['rf_prob']))
    label_map = dict(zip(val_features['window_id'], val_features['label']))

    val_windows = val_windows.copy()
    val_windows['rf_prob'] = val_windows['window_id'].map(prob_map)
    val_windows['window_label'] = val_windows['window_id'].map(label_map)

    print(f"Probability range: {np.nanmin(rf_prob):.4f} - {np.nanmax(rf_prob):.4f}")

    return val_windows, val_features

# =============================================================================
# WINDOW SELECTION
# =============================================================================

def select_validation_windows(val_features, num_windows=6):
    """Select a balanced set of validation windows for plotting."""
    positive_windows = val_features[val_features['label'] == 1]['window_id'].unique().tolist()
    negative_windows = val_features[val_features['label'] == 0]['window_id'].unique().tolist()

    selected = []
    # Take up to half positive windows
    for w in positive_windows[: num_windows // 2]:
        selected.append(w)
    # Fill the rest with negative windows
    for w in negative_windows:
        if len(selected) >= num_windows:
            break
        selected.append(w)

    if len(selected) < num_windows:
        # If not enough windows, pad with remaining positives/negatives
        all_windows = positive_windows + negative_windows
        for w in all_windows:
            if w not in selected:
                selected.append(w)
            if len(selected) >= num_windows:
                break

    return selected[:num_windows]

# =============================================================================
# PLOTTING
# =============================================================================

def plot_validation_windows(val_windows, val_features, selected_windows, output_dir="results"):
    """Plot individual validation windows similar to training window analysis."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, window_id in enumerate(selected_windows):
        ax = axes[idx]

        window_data = val_windows[val_windows['window_id'] == window_id].copy()
        window_features = val_features[val_features['window_id'] == window_id].iloc[0]

        ax2 = ax.twinx()

        # Pressure trace (left axis)
        ax.plot(range(len(window_data)), window_data['PRESSURE'], 'k-', linewidth=2, label='Pressure')
        ax.set_ylabel("Pressure (Pa)", color='k')
        ax.tick_params(axis='y', labelcolor='k')

        # RF probability (right axis) as horizontal line
        rf_prob = window_features['rf_prob']
        ax2.axhline(rf_prob, color='tab:blue', linewidth=3, label=f'RF Prob: {rf_prob:.3f}')
        ax2.set_ylabel("RF Probability", color='tab:blue')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Threshold lines
        for thresh, color, label in [
            (0.45, 'orange', 'High-Recall (0.45)'),
            (0.50, 'tab:blue', 'Threshold (0.5)'),
            (0.90, 'purple', 'High-Precision (0.9)')
        ]:
            ax2.axhline(thresh, color=color, ls='--', alpha=0.6, label=label)

        # Shade precursor/vortex regions using ground-truth columns
        pressure_min = window_data['PRESSURE'].min()
        pressure_max = window_data['PRESSURE'].max()

        if 'gt_detection_win' in window_data.columns:
            mask = window_data['gt_detection_win'].astype(bool)
            if mask.any():
                ax.fill_between(range(len(window_data)), pressure_min, pressure_max,
                                where=mask, color='red', alpha=0.4, label='Precursor')

        if 'gt_fwhm' in window_data.columns:
            mask = window_data['gt_fwhm'].astype(bool)
            if mask.any():
                ax.fill_between(range(len(window_data)), pressure_min, pressure_max,
                                where=mask, color='green', alpha=0.6, label='Vortex')

        label_value = int(window_features['label'])
        correctness = "✅ CORRECT" if (label_value == 1 and rf_prob > 0.5) or (label_value == 0 and rf_prob <= 0.5) else "❌ WRONG"
        correctness_color = 'green' if correctness == "✅ CORRECT" else 'red'

        pressure_drop = window_data['PRESSURE'].iloc[0] - window_data['PRESSURE'].min()
        slope = window_features.get('overall_slope', 0)

        ax.set_title(
            f"Window {window_id}\n"
            f"Validation Label: {label_value} | RF Probability: {rf_prob:.3f} | {correctness}\n"
            f"Pressure Drop: {pressure_drop:.2f} Pa | Slope: {slope:.3f}",
            fontsize=10,
            color=correctness_color
        )

        ax.set_xlabel("Sample Index (0-59)")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    plt.suptitle(
        "Window-Based Validation Analysis: Individual Validation Windows\n"
        "Each window shows 60 pressure samples with Random Forest confidence",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"window_based_validation_analysis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Validation window-based plot saved to: {plot_filename}")

    plt.show()

    return plot_filename

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Starting event-based validation analysis...")

    val_windows, val_features, model = load_validation_data_and_model()
    val_windows_with_probs, val_features_with_probs = add_model_predictions(val_windows, val_features, model)

    selected_windows = select_validation_windows(val_features_with_probs, num_windows=6)
    print(f"Selected validation windows for plotting: {selected_windows}")

    plot_validation_windows(val_windows_with_probs, val_features_with_probs, selected_windows)

    print("\n" + "=" * 70)
    print("EVENT-BASED VALIDATION ANALYSIS COMPLETED")
    print("=" * 70)
    print("- Visualizes representative validation windows (pressure vs model confidence)")
    print("- Highlights precursor (red) and vortex (green) regions")
    print("- Shows Random Forest probability and threshold lines for comparison")
    print("=" * 70)

if __name__ == "__main__":
    main()



