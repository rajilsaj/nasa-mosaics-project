#!/usr/bin/env python3
"""
Mission-Focused Threshold Tuning for Mars Vortex Detection
=========================================================

This script implements the strategic framework for threshold tuning based on
mission-specific operating goals and temporal logic optimization.

Operating Goals:
1. High-Precision Mode: Minimize false alarms (energy conservation)
2. High-Recall Mode: Minimize missed vortices (science priority) 
3. Cost-Based Mode: Optimize weighted cost function
4. Max F1 Mode: Balanced trade-off

Temporal Logic:
- Joint optimization of threshold AND temporal rules
- Runtime decision logic for Mars deployment

Author: ML RF Scientist (Mars Mission Specialist)
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import itertools
import warnings
warnings.filterwarnings('ignore')

class MissionThresholdTuner:
    """Mission-focused threshold tuning with temporal logic."""
    
    def __init__(self, model, val_features_df):
        """
        Initialize mission threshold tuner.
        
        Args:
            model: Trained Random Forest model
            val_features_df: Validation features DataFrame
        """
        self.model = model
        self.val_features_df = val_features_df
        
        # Prepare data
        self.feature_cols = [col for col in val_features_df.columns 
                           if col not in ['window_id', 'start_idx', 'end_idx', 
                                        'start_sclk', 'end_sclk', 'label']]
        
        # Filter out 'Omit' labels and convert to binary
        valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
        valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
        
        # Get features for valid samples only
        self.X_val = valid_df[self.feature_cols].values
        self.y_val = valid_df['label'].values
        
        # Get prediction probabilities
        self.y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        # Add temporal information
        self.window_ids = valid_df['window_id'].values
        self.start_sclks = valid_df['start_sclk'].values
        
        print(f"Mission Threshold Tuner initialized:")
        print(f"  Validation samples: {len(self.y_val):,}")
        print(f"  Class distribution: {np.bincount(self.y_val)}")
        print(f"  Features: {len(self.feature_cols)}")
    
    def high_precision_mode(self, min_precision=0.90):
        """
        High-Precision Mode: Minimize false alarms (energy conservation).
        
        Args:
            min_precision: Minimum precision requirement (default: 90%)
        
        Returns:
            dict: Optimal threshold and metrics
        """
        print(f"\n{'='*60}")
        print(f"HIGH-PRECISION MODE (Energy Conservation)")
        print(f"Target: Precision >= {min_precision:.1%}, Maximize Recall")
        print(f"{'='*60}")
        
        precision, recall, thresholds = precision_recall_curve(self.y_val, self.y_proba)
        
        # Find thresholds that meet precision requirement
        valid_indices = precision >= min_precision
        
        if not np.any(valid_indices):
            print(f"WARNING: No threshold achieves {min_precision:.1%} precision!")
            print("Relaxing precision requirement...")
            min_precision = np.max(precision) * 0.95
            valid_indices = precision >= min_precision
            print(f"Adjusted precision target: {min_precision:.1%}")
        
        if np.any(valid_indices) and len(valid_indices) == len(thresholds):
            # Among valid thresholds, pick the one with highest recall
            valid_recall = recall[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            optimal_idx = np.argmax(valid_recall)
            optimal_threshold = valid_thresholds[optimal_idx]
            optimal_recall = valid_recall[optimal_idx]
            optimal_precision = precision[valid_indices][optimal_idx]
        else:
            # Fallback to precision-recall curve optimum
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
        
        # Calculate detailed metrics
        y_pred = (self.y_proba >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
        
        # Calculate false positive rate per hour
        samples_per_hour = 600  # Assuming ~10 samples per minute
        fp_per_hour = (fp / len(self.y_val)) * samples_per_hour
        
        results = {
            'mode': 'high_precision',
            'threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': 2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall),
            'fp_per_hour': fp_per_hour,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Precision: {optimal_precision:.4f} ({optimal_precision*100:.1f}%)")
        print(f"Recall: {optimal_recall:.4f} ({optimal_recall*100:.1f}%)")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"False positives per hour: {fp_per_hour:.1f}")
        print(f"Energy impact: {fp_per_hour:.1f} false alarms/hour")
        
        return results
    
    def high_recall_mode(self, min_recall=0.90):
        """
        High-Recall Mode: Minimize missed vortices (science priority).
        
        Args:
            min_recall: Minimum recall requirement (default: 90%)
        
        Returns:
            dict: Optimal threshold and metrics
        """
        print(f"\n{'='*60}")
        print(f"HIGH-RECALL MODE (Science Priority)")
        print(f"Target: Recall >= {min_recall:.1%}, Maximize Precision")
        print(f"{'='*60}")
        
        precision, recall, thresholds = precision_recall_curve(self.y_val, self.y_proba)
        
        # Find thresholds that meet recall requirement
        valid_indices = recall >= min_recall
        
        if not np.any(valid_indices):
            print(f"WARNING: No threshold achieves {min_recall:.1%} recall!")
            print("Relaxing recall requirement...")
            min_recall = np.max(recall) * 0.95
            valid_indices = recall >= min_recall
            print(f"Adjusted recall target: {min_recall:.1%}")
        
        if np.any(valid_indices) and len(valid_indices) == len(thresholds):
            # Among valid thresholds, pick the one with highest precision
            valid_precision = precision[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            optimal_idx = np.argmax(valid_precision)
            optimal_threshold = valid_thresholds[optimal_idx]
            optimal_precision = valid_precision[optimal_idx]
            optimal_recall = recall[valid_indices][optimal_idx]
        else:
            # Fallback to precision-recall curve optimum
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_precision = precision[optimal_idx]
            optimal_recall = recall[optimal_idx]
        
        # Calculate detailed metrics
        y_pred = (self.y_proba >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
        
        # Calculate missed vortex rate
        missed_vortex_rate = (fn / len(self.y_val)) * 100
        
        # Calculate false positive rate per hour
        samples_per_hour = 600  # Assuming ~10 samples per minute
        fp_per_hour = (fp / len(self.y_val)) * samples_per_hour
        
        results = {
            'mode': 'high_recall',
            'threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': 2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall),
            'missed_vortex_rate': missed_vortex_rate,
            'fp_per_hour': fp_per_hour,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Precision: {optimal_precision:.4f} ({optimal_precision*100:.1f}%)")
        print(f"Recall: {optimal_recall:.4f} ({optimal_recall*100:.1f}%)")
        print(f"F1-Score: {results['f1_score']:.4f}")
        print(f"Missed vortex rate: {missed_vortex_rate:.2f}%")
        print(f"Science impact: {missed_vortex_rate:.2f}% vortices missed")
        
        return results
    
    def cost_based_mode(self, fn_cost=10, fp_cost=1, power_budget_hours=24):
        """
        Cost-Based Mode: Optimize weighted cost function.
        
        Args:
            fn_cost: Cost of missing a vortex (science cost)
            fp_cost: Cost of false alarm (energy cost)
            power_budget_hours: Daily power budget in hours
        
        Returns:
            dict: Optimal threshold and metrics
        """
        print(f"\n{'='*60}")
        print(f"COST-BASED MODE (Mission Optimization)")
        print(f"Costs: FN={fn_cost}, FP={fp_cost}")
        print(f"Power budget: {power_budget_hours} hours/day")
        print(f"{'='*60}")
        
        # Test different thresholds
        test_thresholds = np.linspace(0.1, 0.95, 85)
        costs = []
        
        for threshold in test_thresholds:
            y_pred = (self.y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            
            # Calculate daily cost
            samples_per_hour = 600
            fp_per_hour = (fp / len(self.y_val)) * samples_per_hour
            fp_per_day = fp_per_hour * power_budget_hours
            
            total_cost = fn * fn_cost + fp_per_day * fp_cost
            costs.append(total_cost)
        
        # Find optimal threshold
        optimal_idx = np.argmin(costs)
        optimal_threshold = test_thresholds[optimal_idx]
        min_cost = costs[optimal_idx]
        
        # Calculate metrics for optimal threshold
        y_pred = (self.y_proba >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate costs
        fp_per_hour = (fp / len(self.y_val)) * samples_per_hour
        fp_per_day = fp_per_hour * power_budget_hours
        science_cost = fn * fn_cost
        energy_cost = fp_per_day * fp_cost
        
        results = {
            'mode': 'cost_based',
            'threshold': optimal_threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_cost': min_cost,
            'science_cost': science_cost,
            'energy_cost': energy_cost,
            'fp_per_hour': fp_per_hour,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Total daily cost: {min_cost:.2f}")
        print(f"  Science cost: {science_cost:.2f} ({fn} missed vortices)")
        print(f"  Energy cost: {energy_cost:.2f} ({fp_per_hour:.1f} FP/hour)")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return results
    
    def max_f1_mode(self):
        """
        Max F1 Mode: Balanced trade-off between precision and recall.
        
        Returns:
            dict: Optimal threshold and metrics
        """
        print(f"\n{'='*60}")
        print(f"MAX F1 MODE (Balanced Trade-off)")
        print(f"Target: Maximize F1-Score (balanced miss/false alarm)")
        print(f"{'='*60}")
        
        precision, recall, thresholds = precision_recall_curve(self.y_val, self.y_proba)
        
        # Calculate F1 scores
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Calculate detailed metrics
        y_pred = (self.y_proba >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
        
        # Calculate rates
        fp_per_hour = (fp / len(self.y_val)) * 600
        missed_rate = (fn / len(self.y_val)) * 100
        
        results = {
            'mode': 'max_f1',
            'threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': optimal_f1,
            'fp_per_hour': fp_per_hour,
            'missed_rate': missed_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        print(f"Precision: {optimal_precision:.4f} ({optimal_precision*100:.1f}%)")
        print(f"Recall: {optimal_recall:.4f} ({optimal_recall*100:.1f}%)")
        print(f"F1-Score: {optimal_f1:.4f}")
        print(f"False positives per hour: {fp_per_hour:.1f}")
        print(f"Missed vortex rate: {missed_rate:.2f}%")
        
        return results
    
    def temporal_logic_optimization(self, base_threshold, window_size=3, min_consecutive=2):
        """
        Optimize temporal logic rules for runtime deployment.
        
        Args:
            base_threshold: Base threshold from single-window optimization
            window_size: Size of temporal window for decision logic
            min_consecutive: Minimum consecutive windows above threshold
        
        Returns:
            dict: Optimal temporal logic and metrics
        """
        print(f"\n{'='*60}")
        print(f"TEMPORAL LOGIC OPTIMIZATION")
        print(f"Base threshold: {base_threshold:.4f}")
        print(f"Window size: {window_size}, Min consecutive: {min_consecutive}")
        print(f"{'='*60}")
        
        # Generate temporal predictions
        temporal_predictions = []
        
        for i in range(len(self.y_proba)):
            # Get probabilities in temporal window
            start_idx = max(0, i - window_size + 1)
            window_probs = self.y_proba[start_idx:i+1]
            
            # Apply temporal logic
            above_threshold = np.sum(window_probs >= base_threshold)
            if above_threshold >= min_consecutive:
                temporal_predictions.append(1)
            else:
                temporal_predictions.append(0)
        
        temporal_predictions = np.array(temporal_predictions)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(self.y_val, temporal_predictions).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        fp_per_hour = (fp / len(self.y_val)) * 600
        missed_rate = (fn / len(self.y_val)) * 100
        
        results = {
            'mode': 'temporal_logic',
            'base_threshold': base_threshold,
            'window_size': window_size,
            'min_consecutive': min_consecutive,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'fp_per_hour': fp_per_hour,
            'missed_rate': missed_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
        
        print(f"Temporal logic: 'Trigger if >={min_consecutive} of last {window_size} windows >= {base_threshold:.4f}'")
        print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
        print(f"Recall: {recall:.4f} ({recall*100:.1f}%)")
        print(f"F1-Score: {f1:.4f}")
        print(f"False positives per hour: {fp_per_hour:.1f}")
        print(f"Missed vortex rate: {missed_rate:.2f}%")
        
        return results
    
    def compare_all_modes(self):
        """Compare all operating modes and provide recommendations."""
        print(f"\n{'='*80}")
        print(f"MISSION THRESHOLD TUNING - COMPREHENSIVE ANALYSIS")
        print(f"{'='*80}")
        
        # Run all modes
        high_prec = self.high_precision_mode(min_precision=0.90)
        high_recall = self.high_recall_mode(min_recall=0.90)
        cost_based = self.cost_based_mode(fn_cost=10, fp_cost=1, power_budget_hours=24)
        max_f1 = self.max_f1_mode()
        
        # Temporal logic optimization using best base threshold
        temporal = self.temporal_logic_optimization(
            base_threshold=high_prec['threshold'], 
            window_size=3, 
            min_consecutive=2
        )
        
        # Compile results
        all_results = {
            'high_precision': high_prec,
            'high_recall': high_recall,
            'cost_based': cost_based,
            'max_f1': max_f1,
            'temporal_logic': temporal
        }
        
        # Create comparison table
        print(f"\n{'='*80}")
        print(f"OPERATING MODE COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = pd.DataFrame({
            'Mode': ['High Precision', 'High Recall', 'Cost-Based', 'Max F1', 'Temporal Logic'],
            'Threshold': [
                f"{all_results['high_precision']['threshold']:.4f}",
                f"{all_results['high_recall']['threshold']:.4f}",
                f"{all_results['cost_based']['threshold']:.4f}",
                f"{all_results['max_f1']['threshold']:.4f}",
                f"{all_results['temporal_logic']['base_threshold']:.4f}"
            ],
            'Precision': [f"{r['precision']:.3f}" for r in all_results.values()],
            'Recall': [f"{r['recall']:.3f}" for r in all_results.values()],
            'F1-Score': [f"{r['f1_score']:.3f}" for r in all_results.values()],
            'FP/Hour': [
                f"{all_results['high_precision']['fp_per_hour']:.1f}",
                f"{all_results['high_recall']['fp_per_hour']:.1f}",
                f"{all_results['cost_based']['fp_per_hour']:.1f}",
                f"{all_results['max_f1']['fp_per_hour']:.1f}",
                f"{all_results['temporal_logic']['fp_per_hour']:.1f}"
            ]
        })
        
        print(comparison_df.to_string(index=False))
        
        # Mission recommendations
        print(f"\n{'='*80}")
        print(f"MISSION RECOMMENDATIONS")
        print(f"{'='*80}")
        
        print(f"\nENERGY-LIMITED OPERATIONS (Power Conservation):")
        print(f"   Recommended: High Precision Mode")
        print(f"   Threshold: {high_prec['threshold']:.4f}")
        print(f"   Benefit: Only {high_prec['fp_per_hour']:.1f} false alarms/hour")
        print(f"   Trade-off: {100-high_prec['recall']*100:.1f}% vortices missed")
        
        print(f"\nSCIENCE-PRIORITY OPERATIONS (Vortex Discovery):")
        print(f"   Recommended: High Recall Mode")
        print(f"   Threshold: {high_recall['threshold']:.4f}")
        print(f"   Benefit: Only {high_recall.get('missed_vortex_rate', 0):.1f}% vortices missed")
        print(f"   Trade-off: {high_recall['fp_per_hour']:.1f} false alarms/hour")
        
        print(f"\nBALANCED MISSION OPERATIONS:")
        print(f"   Recommended: Temporal Logic Mode")
        print(f"   Logic: 'Trigger if >=2 of last 3 windows >= {temporal['base_threshold']:.4f}'")
        print(f"   Benefit: Reduced false alarms while maintaining detection")
        print(f"   Performance: {temporal['precision']:.1%} precision, {temporal['recall']:.1%} recall")
        
        print(f"\nCOST-OPTIMIZED OPERATIONS:")
        print(f"   Recommended: Cost-Based Mode")
        print(f"   Threshold: {cost_based['threshold']:.4f}")
        print(f"   Daily cost: {cost_based['total_cost']:.2f}")
        print(f"   Breakdown: Science={cost_based['science_cost']:.2f}, Energy={cost_based['energy_cost']:.2f}")
        
        # Save results
        results_df = pd.DataFrame(all_results).T
        results_df.to_csv('mission_threshold_results.csv')
        print(f"\nResults saved to: mission_threshold_results.csv")
        
        return all_results

def main():
    """Main mission threshold tuning pipeline."""
    print("="*80)
    print("MISSION-FOCUSED THRESHOLD TUNING - MARS VORTEX DETECTION")
    print("="*80)
    print("Strategic framework for Mars rover deployment optimization")
    print("="*80)
    
    # Load data and train model
    print("\nLoading training features and training model...")
    train_features_df = pd.read_csv("train_features.csv")
    
    feature_cols = [col for col in train_features_df.columns if col not in ['window_id', 'label', 'event_sclk', 'split']]
    X_train = train_features_df[feature_cols].values
    y_train = train_features_df['label'].values
    
    # Train Random Forest model
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
    rf_model.fit(X_train, y_train)
    print("Model trained successfully!")
    
    # Load validation features
    print("\nLoading validation features...")
    val_features_df = pd.read_csv("val_sliding_features.csv")
    
    # Initialize mission threshold tuner
    tuner = MissionThresholdTuner(rf_model, val_features_df)
    
    # Run comprehensive analysis
    results = tuner.compare_all_modes()
    
    print(f"\n{'='*80}")
    print(f"MISSION THRESHOLD TUNING COMPLETED")
    print(f"{'='*80}")
    print(f"Ready for Mars deployment with mission-specific optimization!")
    
    return results

if __name__ == "__main__":
    results = main()
