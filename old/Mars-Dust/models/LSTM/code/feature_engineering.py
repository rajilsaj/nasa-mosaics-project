# feature_engineering.py
"""
Feature extraction and post-processing filter functions for LSTM vortex detection.
"""
import numpy as np

def apply_postprocessing_filters(y_pred, X_test_pressure, y_pred_proba, thresholds, lookahead=5):
    """
    Apply post-processing filters to LSTM detections.
    Args:
        y_pred: np.ndarray, binary predictions (0/1)
        X_test_pressure: np.ndarray, shape (n_samples, window_size), pressure windows
        y_pred_proba: np.ndarray, predicted probabilities
        thresholds: dict, e.g. {
            'sharp_drop': -0.35,
            'total_drop_after_initial': 0.1,
            'avg_slope_after_sharpest': -0.3
        }
        lookahead: int, number of points after sharpest drop for avg slope
    Returns:
        y_pred_filtered: np.ndarray, filtered predictions
    """
    y_pred_filtered = y_pred.copy()
    for i, pred in enumerate(y_pred):
        if pred != 1:
            continue
        p = X_test_pressure[i]
        diffs = np.diff(p)
        # 1. Sharpest drop (min slope)
        if 'sharp_drop' in thresholds:
            min_slope = np.min(diffs)
            if min_slope < thresholds['sharp_drop']:
                y_pred_filtered[i] = 0
                continue
        # 2. Total drop after initial drop
        if 'total_drop_after_initial' in thresholds:
            if np.any(diffs < 0):
                initial_drop_idx = np.argmax(diffs < 0)
                drop_after_initial = p[-1] - p[initial_drop_idx]
            else:
                drop_after_initial = 0
            if not (drop_after_initial < thresholds['total_drop_after_initial']):
                y_pred_filtered[i] = 0
                continue
        # 3. Avg slope after sharpest drop
        if 'avg_slope_after_sharpest' in thresholds:
            sharp_idx = np.argmin(diffs)
            after = p[sharp_idx+1:sharp_idx+1+lookahead]
            before = p[sharp_idx]
            if len(after) > 0:
                avg_slope = (after[-1] - before) / len(after)
            else:
                avg_slope = 0
            if not (avg_slope > thresholds['avg_slope_after_sharpest']):
                y_pred_filtered[i] = 0
                continue
        # (Add more filters here as needed)
    return y_pred_filtered 