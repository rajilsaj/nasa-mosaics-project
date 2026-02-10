#!/usr/bin/env python3
"""
Threshold Calibration for Mars Vortex Detection
==============================================

This script performs systematic threshold calibration on the validation set
to optimize the decision threshold for deployment scenarios.

Approaches:
1. ROC-based threshold selection (Youden's J statistic)
2. Precision-Recall curve optimization
3. Cost-sensitive threshold selection
4. Mission-specific threshold selection

Author: ML RF Scientist
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score
)
from scipy.optimize import minimize_scalar
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ThresholdCalibrator:
    """Threshold calibration for Mars vortex detection."""
    
    def __init__(self, model, val_features_df, mission_requirements=None):
        """
        Initialize calibrator.
        
        Args:
            model: Trained Random Forest model
            val_features_df: Validation features DataFrame
            mission_requirements: Dict with mission-specific constraints
        """
        self.model = model
        self.val_features_df = val_features_df
        self.mission_requirements = mission_requirements or {
            'max_false_positives_per_hour': 10,  # Max acceptable false positives
            'min_recall_threshold': 0.3,        # Minimum recall required
            'preferred_precision': 0.8          # Preferred precision level
        }
        
        # Prepare data
        self.feature_cols = [col for col in val_features_df.columns 
                           if col not in ['window_id', 'start_idx', 'end_idx', 
                                        'start_sclk', 'end_sclk', 'label']]
        self.X_val = val_features_df[self.feature_cols].values
        
        # Filter out 'Omit' labels and convert to binary
        valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
        valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
        self.y_val = valid_df['label'].values
        
        # Get prediction probabilities
        self.y_proba = self.model.predict_proba(self.X_val)[:, 1]
        
        print(f"Validation data prepared:")
        print(f"  Total samples: {len(self.y_val):,}")
        print(f"  Class distribution: {np.bincount(self.y_val)}")
        print(f"  Features: {len(self.feature_cols)}")
    
    def find_optimal_thresholds(self):
        """Find optimal thresholds using multiple criteria."""
        print("\n" + "="*70)
        print("THRESHOLD CALIBRATION - MULTIPLE CRITERIA")
        print("="*70)
        
        # 1. ROC-based threshold (Youden's J statistic)
        fpr, tpr, roc_thresholds = roc_curve(self.y_val, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Youden's J = TPR - FPR (maximizes true positive rate while minimizing false positive rate)
        youden_j = tpr - fpr
        optimal_roc_idx = np.argmax(youden_j)
        threshold_roc = roc_thresholds[optimal_roc_idx]
        
        print(f"\n1. ROC-Based Threshold (Youden's J):")
        print(f"   Threshold: {threshold_roc:.4f}")
        print(f"   ROC AUC: {roc_auc:.4f}")
        
        # 2. Precision-Recall based threshold
        precision, recall, pr_thresholds = precision_recall_curve(self.y_val, self.y_proba)
        pr_auc = average_precision_score(self.y_val, self.y_proba)
        
        # Find threshold that maximizes F1-score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_pr_idx = np.argmax(f1_scores)
        threshold_pr = pr_thresholds[optimal_pr_idx]
        
        print(f"\n2. Precision-Recall Based Threshold (F1-max):")
        print(f"   Threshold: {threshold_pr:.4f}")
        print(f"   PR AUC: {pr_auc:.4f}")
        
        # 3. Cost-sensitive threshold
        # Assume cost of false positive = 1, cost of false negative = 10 (missed vortex is expensive!)
        fp_cost, fn_cost = 1, 10
        
        def cost_function(threshold):
            y_pred = (self.y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            total_cost = fp * fp_cost + fn * fn_cost
            return total_cost
        
        # Find threshold that minimizes cost
        result = minimize_scalar(cost_function, bounds=(0, 1), method='bounded')
        threshold_cost = result.x
        
        print(f"\n3. Cost-Sensitive Threshold (FP cost=1, FN cost=10):")
        print(f"   Threshold: {threshold_cost:.4f}")
        print(f"   Expected cost: {result.fun:.2f}")
        
        # 4. Mission-specific threshold
        # Find threshold that meets minimum recall requirement while maximizing precision
        valid_recall_indices = recall >= self.mission_requirements['min_recall_threshold']
        if np.any(valid_recall_indices):
            valid_precision = precision[valid_recall_indices]
            valid_thresholds = pr_thresholds[valid_recall_indices]
            optimal_mission_idx = np.argmax(valid_precision)
            threshold_mission = valid_thresholds[optimal_mission_idx]
        else:
            threshold_mission = threshold_pr  # Fallback to F1-optimal
        
        print(f"\n4. Mission-Specific Threshold (min recall={self.mission_requirements['min_recall_threshold']}):")
        print(f"   Threshold: {threshold_mission:.4f}")
        
        # Store all thresholds
        self.thresholds = {
            'roc_youden': threshold_roc,
            'pr_f1': threshold_pr,
            'cost_sensitive': threshold_cost,
            'mission_specific': threshold_mission
        }
        
        return self.thresholds
    
    def evaluate_thresholds(self):
        """Evaluate all thresholds on validation set."""
        print(f"\n" + "="*70)
        print("THRESHOLD EVALUATION ON VALIDATION SET")
        print("="*70)
        
        results = {}
        
        for name, threshold in self.thresholds.items():
            # Make predictions
            y_pred = (self.y_proba >= threshold).astype(int)
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Calculate false positive rate per hour (assuming ~10 samples per minute)
            samples_per_hour = 60 * 10  # 600 samples per hour
            fp_per_hour = (fp / len(self.y_val)) * samples_per_hour
            
            results[name] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'fp_per_hour': fp_per_hour
            }
            
            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  TP: {tp:4d}  FP: {fp:4d}  TN: {tn:5d}  FN: {fn:3d}")
            print(f"  FP per hour: {fp_per_hour:.1f}")
        
        self.results = results
        return results
    
    def recommend_threshold(self):
        """Recommend best threshold based on mission requirements."""
        print(f"\n" + "="*70)
        print("THRESHOLD RECOMMENDATION")
        print("="*70)
        
        # Filter thresholds that meet minimum requirements
        valid_thresholds = {}
        for name, metrics in self.results.items():
            if (metrics['recall'] >= self.mission_requirements['min_recall_threshold'] and
                metrics['fp_per_hour'] <= self.mission_requirements['max_false_positives_per_hour']):
                valid_thresholds[name] = metrics
        
        if not valid_thresholds:
            print("WARNING: No threshold meets all mission requirements!")
            print("Consider relaxing constraints or improving model.")
            # Use the one with highest F1 as fallback
            best_threshold = max(self.results.items(), key=lambda x: x[1]['f1_score'])
            recommended_name = best_threshold[0]
            recommended_threshold = best_threshold[1]['threshold']
        else:
            # Among valid thresholds, pick the one with highest precision
            best_threshold = max(valid_thresholds.items(), key=lambda x: x[1]['precision'])
            recommended_name = best_threshold[0]
            recommended_threshold = best_threshold[1]['threshold']
        
        print(f"Recommended threshold: {recommended_name.replace('_', ' ').title()}")
        print(f"Threshold value: {recommended_threshold:.4f}")
        
        recommended_metrics = self.results[recommended_name]
        print(f"\nExpected performance:")
        print(f"  Precision: {recommended_metrics['precision']:.4f}")
        print(f"  Recall:    {recommended_metrics['recall']:.4f}")
        print(f"  F1-Score:  {recommended_metrics['f1_score']:.4f}")
        print(f"  FP per hour: {recommended_metrics['fp_per_hour']:.1f}")
        
        self.recommended_threshold = recommended_threshold
        self.recommended_name = recommended_name
        
        return recommended_threshold, recommended_name
    
    def plot_calibration_curves(self, save_plots=True):
        """Plot calibration curves and threshold analysis."""
        print(f"\nGenerating calibration plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Threshold Calibration Analysis - Mars Vortex Detection', fontsize=16)
        
        # 1. ROC Curve
        fpr, tpr, roc_thresholds = roc_curve(self.y_val, self.y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, pr_thresholds = precision_recall_curve(self.y_val, self.y_proba)
        pr_auc = average_precision_score(self.y_val, self.y_proba)
        
        axes[0, 1].plot(recall, precision, color='red', lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="lower left")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Threshold vs Metrics
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for thresh in thresholds:
            y_pred = (self.y_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_val, y_pred).ravel()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
        
        axes[1, 0].plot(thresholds, precisions, label='Precision', color='blue')
        axes[1, 0].plot(thresholds, recalls, label='Recall', color='green')
        axes[1, 0].plot(thresholds, f1_scores, label='F1-Score', color='red')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Metrics vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mark optimal thresholds
        for name, threshold in self.thresholds.items():
            axes[1, 0].axvline(x=threshold, linestyle='--', alpha=0.7,
                             label=f'{name.replace("_", " ").title()}')
        
        # 4. Performance Comparison
        threshold_names = list(self.results.keys())
        precisions = [self.results[name]['precision'] for name in threshold_names]
        recalls = [self.results[name]['recall'] for name in threshold_names]
        
        x = np.arange(len(threshold_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
        axes[1, 1].bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
        
        axes[1, 1].set_xlabel('Threshold Method')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([name.replace('_', ' ').title() for name in threshold_names], 
                                  rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('threshold_calibration_analysis.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: threshold_calibration_analysis.png")
        
        plt.show()
    
    def save_results(self, output_file='threshold_calibration_results.csv'):
        """Save calibration results to CSV."""
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(output_file)
        print(f"  Saved: {output_file}")
        
        return results_df

def main():
    """Main threshold calibration pipeline."""
    parser = argparse.ArgumentParser(description='Threshold calibration for Mars vortex detection')
    parser.add_argument('--model_file', type=str, help='Path to trained model file')
    parser.add_argument('--val_features', type=str, default='val_sliding_features.csv',
                       help='Validation features CSV file')
    parser.add_argument('--min_recall', type=float, default=0.3,
                       help='Minimum recall requirement')
    parser.add_argument('--max_fp_per_hour', type=float, default=10.0,
                       help='Maximum false positives per hour')
    args = parser.parse_args()
    
    print("="*70)
    print("THRESHOLD CALIBRATION - MARS VORTEX DETECTION")
    print("="*70)
    
    # Load data
    print(f"\nLoading validation features from: {args.val_features}")
    val_features_df = pd.read_csv(args.val_features)
    print(f"  Loaded {len(val_features_df):,} validation samples")
    
    # Load model (simplified - you might need to adjust based on your model format)
    print(f"\nLoading trained Random Forest model...")
    # For now, we'll assume the model is already trained in the evaluation script
    # You might need to load from a pickle file here
    
    # Mission requirements
    mission_requirements = {
        'min_recall_threshold': args.min_recall,
        'max_false_positives_per_hour': args.max_fp_per_hour,
        'preferred_precision': 0.8
    }
    
    print(f"\nMission requirements:")
    print(f"  Minimum recall: {mission_requirements['min_recall_threshold']}")
    print(f"  Max FP per hour: {mission_requirements['max_false_positives_per_hour']}")
    
    # Initialize calibrator
    calibrator = ThresholdCalibrator(None, val_features_df, mission_requirements)
    
    # Find optimal thresholds
    thresholds = calibrator.find_optimal_thresholds()
    
    # Evaluate thresholds
    results = calibrator.evaluate_thresholds()
    
    # Get recommendation
    recommended_threshold, recommended_name = calibrator.recommend_threshold()
    
    # Save results
    results_df = calibrator.save_results()
    
    print(f"\n" + "="*70)
    print("THRESHOLD CALIBRATION COMPLETED")
    print("="*70)
    print(f"Recommended threshold: {recommended_threshold:.4f}")
    print(f"Method: {recommended_name.replace('_', ' ').title()}")
    
    return recommended_threshold, results_df

if __name__ == "__main__":
    recommended_threshold, results = main()


