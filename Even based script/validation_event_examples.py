#!/usr/bin/env python3
"""
Validation Event Examples
-------------------------
Generate concise plots for three representative validation windows:
- True Positive (precursor + RF spike above threshold)
- Miss (precursor but probability below threshold)
- False Positive (no precursor but probability above threshold)

The script uses the natural validation distribution (val_sliding_features.csv
and ml_val.csv) and the trained Random Forest model.
"""

import argparse
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_PATH = "models/improved_rf_vortex_detector_20251010_114031.pkl"
ML_VAL_FILE = "datasets/temporal_splits/ml_val.csv"
SLIDING_FEATURES_FILE = "datasets/val_sliding_features.csv"
OUTPUT_DIR = "results"
THRESHOLD_DEFAULT = 0.45


def load_data():
    ml_df = pd.read_csv(ML_VAL_FILE)
    bool_map = {True: True, False: False, 'True': True, 'False': False, 1: True, 0: False}
    for col in ['gt_4xfwhm', 'gt_detection_win', 'gt_fwhm']:
        if col in ml_df.columns:
            ml_df[col] = ml_df[col].map(bool_map).fillna(False)
    sliding_df = pd.read_csv(SLIDING_FEATURES_FILE)
    return ml_df, sliding_df


def prepare_predictions(sliding_df):
    model = joblib.load(MODEL_PATH)
    feature_cols = [
        col for col in sliding_df.columns
        if col not in ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label']
    ]

    valid_df = sliding_df[sliding_df['label'] != 'Omit'].copy()
    X = valid_df[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    valid_df['probability'] = probs
    valid_df['is_positive'] = valid_df['label'] == 'True'
    return valid_df


def pick_examples(valid_df, threshold, num_examples):
    examples = {}

    tp_df = valid_df[(valid_df['is_positive']) & (valid_df['probability'] >= threshold)]
    miss_df = valid_df[(valid_df['is_positive']) & (valid_df['probability'] < threshold)]
    fp_df = valid_df[(~valid_df['is_positive']) & (valid_df['probability'] >= threshold)]

    if not tp_df.empty:
        examples['True Positive'] = tp_df.sort_values('probability', ascending=False).head(num_examples)
    if not miss_df.empty:
        examples['Miss'] = miss_df.sort_values('probability', ascending=False).head(num_examples)
    if not fp_df.empty:
        examples['False Positive'] = fp_df.sort_values('probability', ascending=False).head(num_examples)

    return examples


def plot_examples(ml_df, examples, threshold, num_examples):
    if not examples:
        print("No matching examples found for the specified threshold.")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    example_order = ['True Positive', 'Miss', 'False Positive']
    data_sequence = [(title, examples[title]) for title in example_order if title in examples]

    max_cols = max(len(df) for _, df in data_sequence)
    fig, axes = plt.subplots(len(data_sequence), max_cols,
                             figsize=(5.5 * max_cols, 4.5 * len(data_sequence)), squeeze=False)

    for row_idx, (title, df_examples) in enumerate(data_sequence):
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(df_examples):
                ax.axis('off')
                continue

            row = df_examples.iloc[col_idx]
            start_idx = int(row['start_idx'])
            end_idx = int(row['end_idx'])

            segment = ml_df.iloc[start_idx:end_idx + 1].copy()
            x = np.arange(start_idx, end_idx + 1)

            pressure_min = segment['PRESSURE'].min()
            pressure_max = segment['PRESSURE'].max()

            ax.plot(x, segment['PRESSURE'], 'k-', linewidth=2, label='Pressure')

            if 'gt_4xfwhm' in segment.columns and segment['gt_4xfwhm'].any():
                ax.fill_between(
                    x,
                    pressure_min,
                    pressure_max,
                    where=segment['gt_4xfwhm'].astype(bool),
                    color='lightgray',
                    alpha=0.2,
                    label='4xFWHM'
                )

            if 'gt_detection_win' in segment.columns and segment['gt_detection_win'].any():
                ax.fill_between(
                    x,
                    pressure_min,
                    pressure_max,
                    where=segment['gt_detection_win'].astype(bool),
                    color='red',
                    alpha=0.35,
                    label='Precursor'
                )

            if 'gt_fwhm' in segment.columns and segment['gt_fwhm'].any():
                ax.fill_between(
                    x,
                    pressure_min,
                    pressure_max,
                    where=segment['gt_fwhm'].astype(bool),
                    color='green',
                    alpha=0.45,
                    label='Vortex'
                )

            ax.set_xlabel('Global Sample Index')
            ax.set_ylabel('Pressure (Pa)')

            ax2 = ax.twinx()
            prob = row['probability']
            ax2.plot(x, np.full_like(x, prob), color='tab:blue', linewidth=2, label='RF Probability')
            ax2.axhline(threshold, color='orange', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold:.2f})')
            ax2.axhline(0.90, color='purple', linestyle='--', linewidth=1, alpha=0.6, label='High-Precision (0.90)')
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Predicted Probability')

            label_str = 'Positive' if row['is_positive'] else 'Negative'
            status = '✅' if (row['is_positive'] and prob >= threshold) or ((not row['is_positive']) and prob < threshold) else '❌'
            ax.set_title(
                f"{title} #{col_idx+1}\nLabel: {label_str} | Prob: {prob:.3f} | {status}\n"
                f"Pressure Drop: {row['pressure_drop']:.2f} Pa | Slope: {row['overall_slope']:.3f}",
                fontsize=11
            )

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

    plt.suptitle(
        f'Validation Window Examples (Threshold={threshold:.2f})',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = os.path.join(OUTPUT_DIR, f"validation_event_examples_{threshold:.2f}_{timestamp}.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Saved validation event examples to: {outfile}")
    return outfile


def main():
    parser = argparse.ArgumentParser(description='Generate representative validation window plots.')
    parser.add_argument('--threshold', type=float, default=THRESHOLD_DEFAULT,
                        help='Decision threshold for classifying predictions (default: 0.45)')
    parser.add_argument('--num_examples', type=int, default=1,
                        help='Number of examples to display per category (default: 1)')
    args = parser.parse_args()

    ml_df, sliding_df = load_data()
    valid_df = prepare_predictions(sliding_df)
    examples = pick_examples(valid_df, args.threshold, args.num_examples)
    plot_examples(ml_df, examples, args.threshold, args.num_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=THRESHOLD_DEFAULT,
                        help='Decision threshold for classifying predictions (default: 0.45)')
    parser.add_argument('--num_examples', type=int, default=1,
                        help='Number of examples to display per category (default: 1)')
    args = parser.parse_args()

    ml_df, sliding_df = load_data()
    valid_df = prepare_predictions(sliding_df)
    example_dict = pick_examples(valid_df, args.threshold, args.num_examples)
    plot_examples(ml_df, example_dict, args.threshold, args.num_examples)
