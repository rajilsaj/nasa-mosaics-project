"""
Pattern Analysis Script

Analyze statistical differences between true positive and false positive patterns
to find simple discriminative features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_pattern_statistics(successful_patterns, failed_patterns):
    """Analyze statistical differences between TP and FP patterns."""
    
    print("=== PATTERN STATISTICS ANALYSIS ===")
    
    # Convert to numpy arrays if they aren't already
    successful_patterns = np.array(successful_patterns)
    failed_patterns = np.array(failed_patterns)
    
    print(f"Successful patterns: {len(successful_patterns)}")
    print(f"Failed patterns: {len(failed_patterns)}")
    
    # Calculate various statistics for each pattern
    stats = {}
    
    for pattern_set, name in [(successful_patterns, 'successful'), (failed_patterns, 'failed')]:
        if len(pattern_set) == 0:
            continue
            
        # Basic pressure statistics
        mean_pressure = np.mean(pattern_set, axis=1)
        median_pressure = np.median(pattern_set, axis=1)
        pressure_std = np.std(pattern_set, axis=1)
        
        # NORMALIZED pressure statistics (relative to each window's own baseline)
        # This handles different pressure baselines per day
        normalized_median_pressure = []
        normalized_pressure_std = []
        normalized_slope_std = []
        
        for pattern in pattern_set:
            # Normalize relative to this window's own baseline
            window_mean = np.mean(pattern)
            window_std = np.std(pattern)
            
            # Normalized median (relative to window mean)
            normalized_median = np.median(pattern) - window_mean
            normalized_median_pressure.append(normalized_median)
            
            # Normalized pressure std (relative to window)
            normalized_pressure_std.append(window_std)
            
            # Normalized slope std (relative to window pressure variation)
            slopes = np.diff(pattern)
            normalized_slope_std.append(np.std(slopes) / (window_std + 1e-8))
        
        normalized_median_pressure = np.array(normalized_median_pressure)
        normalized_pressure_std = np.array(normalized_pressure_std)
        normalized_slope_std = np.array(normalized_slope_std)
        
        # Slope statistics
        slopes = np.diff(pattern_set, axis=1)
        mean_slope = np.mean(slopes, axis=1)
        slope_std = np.std(slopes, axis=1)
        min_slope = np.min(slopes, axis=1)
        max_slope = np.max(slopes, axis=1)
        
        # Drop statistics
        total_drop = pattern_set[:, -1] - pattern_set[:, 0]
        max_drop = np.min(slopes, axis=1)  # Most negative slope
        
        # Pattern consistency
        autocorr = []
        for pattern in pattern_set:
            if len(pattern) > 1:
                corr = np.corrcoef(pattern[:-1], pattern[1:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
            else:
                autocorr.append(0)
        autocorr = np.array(autocorr)
        
        # Option 1: Mean Slope of Min-Max Normalized Series
        mean_slope_norm = []
        for pattern in pattern_set:
            y_norm = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern) + 1e-8)
            mean_slope_norm.append(np.mean(np.diff(y_norm)))
        mean_slope_norm = np.array(mean_slope_norm)

        # Option 3: Mean Relative Slope (Percent Change)
        mean_rel_slope = []
        for pattern in pattern_set:
            rel_slope = np.diff(pattern) / (pattern[:-1] + 1e-8)
            mean_rel_slope.append(np.mean(rel_slope))
        mean_rel_slope = np.array(mean_rel_slope)

        # Store statistics
        stats[name] = {
            'mean_pressure': mean_pressure,
            'median_pressure': median_pressure,
            'pressure_std': pressure_std,
            'normalized_median_pressure': normalized_median_pressure,
            'normalized_pressure_std': normalized_pressure_std,
            'normalized_slope_std': normalized_slope_std,
            'mean_slope': mean_slope,
            'slope_std': slope_std,
            'min_slope': min_slope,
            'max_slope': max_slope,
            'total_drop': total_drop,
            'max_drop': max_drop,
            'autocorr': autocorr,
            'mean_slope_norm': mean_slope_norm,
            'mean_rel_slope': mean_rel_slope
        }
    
    # Compare statistics
    if 'successful' in stats and 'failed' in stats:
        print("\n=== STATISTICAL COMPARISONS ===")
        
        for stat_name in [
            'normalized_median_pressure', 'normalized_pressure_std', 'normalized_slope_std',
            'mean_slope', 'slope_std', 'autocorr',
            'mean_slope_norm', 'mean_rel_slope']:
            
            successful_vals = stats['successful'][stat_name]
            failed_vals = stats['failed'][stat_name]
            
            successful_mean = np.mean(successful_vals)
            failed_mean = np.mean(failed_vals)
            successful_std = np.std(successful_vals)
            failed_std = np.std(failed_vals)
            
            # Calculate separation (how well this feature discriminates)
            separation = abs(successful_mean - failed_mean) / (successful_std + failed_std + 1e-8)
            
            print(f"{stat_name:25s}: Successful={successful_mean:8.3f}±{successful_std:6.3f}, "
                  f"Failed={failed_mean:8.3f}±{failed_std:6.3f}, Separation={separation:6.3f}")
    
    return stats

def find_best_thresholds(stats):
    """Find optimal thresholds for discriminating between TP and FP patterns."""
    
    if 'successful' not in stats or 'failed' not in stats:
        return {}
    
    print("\n=== OPTIMAL THRESHOLDS ===")
    
    thresholds = {}
    
    for stat_name in [
        'normalized_median_pressure', 'normalized_pressure_std', 'normalized_slope_std',
        'mean_slope', 'slope_std', 'autocorr',
        'mean_slope_norm', 'mean_rel_slope']:
        
        successful_vals = stats['successful'][stat_name]
        failed_vals = stats['failed'][stat_name]
        
        # Find threshold that maximizes separation
        all_vals = np.concatenate([successful_vals, failed_vals])
        all_labels = np.concatenate([np.ones(len(successful_vals)), np.zeros(len(failed_vals))])
        
        best_threshold = None
        best_f1 = 0
        
        for threshold in np.percentile(all_vals, np.arange(5, 95, 5)):
            # Try both directions (greater than and less than)
            for direction in ['gt', 'lt']:
                if direction == 'gt':
                    pred = (all_vals > threshold).astype(int)
                else:
                    pred = (all_vals < threshold).astype(int)
                
                # Calculate F1 score
                tp = np.sum((pred == 1) & (all_labels == 1))
                fp = np.sum((pred == 1) & (all_labels == 0))
                fn = np.sum((pred == 0) & (all_labels == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_direction = direction
        
        if best_threshold is not None:
            thresholds[stat_name] = {
                'threshold': best_threshold,
                'direction': best_direction,
                'f1_score': best_f1
            }
            print(f"{stat_name:15s}: threshold={best_threshold:8.3f} ({best_direction}), F1={best_f1:.3f}")
    
    return thresholds

if __name__ == "__main__":
    # This would be called with actual pattern data
    print("Pattern analysis script - import and use with actual pattern data") 