"""
Event-based evaluation for vortex detection.

This module implements realistic event-based metrics for vortex detection:
- Single detection during gt_detection_win = success for entire event
- Detection persists until end of gt_fwhm
- More realistic than point-wise evaluation
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score


def extract_vortex_events(gt_detection_win: np.ndarray, gt_fwhm: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract vortex events from ground truth.
    
    Args:
        gt_detection_win: Array of detection window labels (1 = precursor)
        gt_fwhm: Array of FWHM labels (1 = during vortex)
        
    Returns:
        List of (start_idx, end_idx) tuples for each vortex event
    """
    events = []
    in_event = False
    start_idx = None
    
    for i in range(len(gt_detection_win)):
        # Start of event: gt_detection_win becomes 1
        if gt_detection_win[i] == 1 and not in_event:
            start_idx = i
            in_event = True
        
        # End of event: gt_fwhm becomes 0 (after being 1)
        elif in_event and gt_fwhm[i] == 0:
            # Check if we've been in the vortex phase
            if any(gt_fwhm[start_idx:i] == 1):
                events.append((start_idx, i - 1))
            in_event = False
            start_idx = None
    
    # Handle case where event extends to end of data
    if in_event:
        if any(gt_fwhm[start_idx:] == 1):
            events.append((start_idx, len(gt_detection_win) - 1))
    
    return events


def evaluate_vortex_detection(predictions: np.ndarray, gt_detection_win: np.ndarray, 
                            gt_fwhm: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate vortex detection with simple latch-on logic.
    
    Detection Logic:
    - If model predicts positive AT ANY POINT during gt_detection_win = TP for entire event
    - Once triggered, count all subsequent points as TP until end of gt_fwhm
    - Simple point-wise evaluation with event-level success criteria
    
    Args:
        predictions: Model predictions (0 or 1)
        gt_detection_win: Ground truth detection window labels
        gt_fwhm: Ground truth FWHM labels
        verbose: Print detailed results
        
    Returns:
        Dictionary with event-based metrics
    """
    # Extract vortex events
    events = extract_vortex_events(gt_detection_win, gt_fwhm)
    
    if verbose:
        print(f"\nEvent-based Evaluation:")
        print(f"Total vortex events found: {len(events)}")
        if len(events) > 0:
            print(f"First few events: {events[:5]}")
    
    # SIMPLE LATCH-ON APPROACH: While loop implementation
    tp_points = 0
    fp_points = 0
    fn_points = 0
    tn_points = 0
    
    i = 0
    while i < len(predictions):
        # Check if we're in a vortex event (gt_detection_win or gt_fwhm is true)
        in_vortex = gt_detection_win[i] == 1 or gt_fwhm[i] == 1
        
        if in_vortex:
            # We're in a vortex event
            # Check if we have a prediction during gt_detection_win period
            if gt_detection_win[i] == 1 and predictions[i] == 1:
                # Initial TP - trigger the latch-on
                tp_points += 1
                i += 1
                
                # Continue counting as TP while in vortex event
                while i < len(predictions) and (gt_detection_win[i] == 1 or gt_fwhm[i] == 1):
                    tp_points += 1
                    i += 1
            else:
                # No trigger, count as FN
                fn_points += 1
                i += 1
        else:
            # Outside vortex event
            if predictions[i] == 1:
                fp_points += 1
            else:
                tn_points += 1
            i += 1
    
    # Calculate metrics
    total_points = len(predictions)
    event_precision = tp_points / (tp_points + fp_points) if (tp_points + fp_points) > 0 else 0
    event_recall = tp_points / (tp_points + fn_points) if (tp_points + fn_points) > 0 else 0
    event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0
    
    # Calculate point-wise metrics for comparison
    point_precision = precision_score(gt_detection_win, predictions, zero_division=0)
    point_recall = recall_score(gt_detection_win, predictions, zero_division=0)
    point_f1 = f1_score(gt_detection_win, predictions, zero_division=0)
    
    results = {
        'event_metrics': {
            'precision': event_precision,
            'recall': event_recall,
            'f1': event_f1,
            'tp_points': tp_points,
            'fp_points': fp_points,
            'fn_points': fn_points,
            'tn_points': tn_points,
            'total_points': total_points
        },
        'point_metrics': {
            'precision': point_precision,
            'recall': point_recall,
            'f1': point_f1
        },
        'events': events
    }
    
    if verbose:
        print(f"\nEvent-based Results (Latch-on Logic):")
        print(f"  Event Precision: {event_precision:.4f}")
        print(f"  Event Recall:    {event_recall:.4f}")
        print(f"  Event F1-Score:  {event_f1:.4f}")
        print(f"  TP Points:       {tp_points}")
        print(f"  FP Points:       {fp_points}")
        print(f"  FN Points:       {fn_points}")
        print(f"  TN Points:       {tn_points}")
        
        print(f"\nPoint-wise Comparison:")
        print(f"  Point Precision: {point_precision:.4f}")
        print(f"  Point Recall:    {point_recall:.4f}")
        print(f"  Point F1-Score:  {point_f1:.4f}")
    
    return results





def analyze_detection_timing(predictions: np.ndarray, events: List[Tuple[int, int]], 
                           gt_detection_win: np.ndarray, gt_fwhm: np.ndarray) -> Dict[str, Any]:
    """
    Analyze timing of detections relative to vortex events.
    
    Args:
        predictions: Model predictions
        events: List of vortex events
        gt_detection_win: Ground truth detection window labels
        gt_fwhm: Ground truth FWHM labels
        
    Returns:
        Dictionary with timing analysis
    """
    detection_timings = []
    
    for start_idx, end_idx in events:
        # Find first prediction during gt_detection_win period
        event_detection_win = gt_detection_win[start_idx:end_idx+1]
        event_predictions = predictions[start_idx:end_idx+1]
        
        # Find indices where gt_detection_win == 1 (precursor period)
        precursor_indices = np.where(event_detection_win == 1)[0]
        
        # Find first prediction during precursor period
        first_detection_idx = None
        for idx in precursor_indices:
            if event_predictions[idx] == 1:
                first_detection_idx = start_idx + idx
                break
        
        if first_detection_idx is not None:
            # Calculate timing relative to event start
            timing = first_detection_idx - start_idx
            detection_timings.append(timing)
    
    if detection_timings:
        timing_stats = {
            'mean_timing': np.mean(detection_timings),
            'median_timing': np.median(detection_timings),
            'min_timing': np.min(detection_timings),
            'max_timing': np.max(detection_timings),
            'std_timing': np.std(detection_timings),
            'timings': detection_timings
        }
    else:
        timing_stats = {
            'mean_timing': 0,
            'median_timing': 0,
            'min_timing': 0,
            'max_timing': 0,
            'std_timing': 0,
            'timings': []
        }
    
    return timing_stats


def main():
    """Example usage of event-based evaluation."""
    print("Event-based evaluation module loaded successfully!")
    print("Use evaluate_vortex_detection() to analyze your predictions.")


if __name__ == "__main__":
    main() 