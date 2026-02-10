#!/usr/bin/env python3
"""
Simple Threshold Tuning for Mars Vortex Detection
================================================

This script performs threshold calibration on the validation set
using the trained Random Forest model from sliding_window_evaluation.py.

Key Approaches:
1. ROC-based threshold (Youden's J statistic) - balances TPR and FPR
2. F1-score optimization - balances precision and recall
3. Cost-sensitive threshold - considers mission costs
4. Mission-specific threshold - meets deployment requirements

Usage:
    python simple_threshold_tuning.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score
)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data_and_model():
    """Load validation features and train model."""
    print("="*70)
    print("LOADING DATA AND TRAINING MODEL")
    print("="*70)
    
    # Load training features to train model
    print("Loading training features...")
    train_features_df = pd.read_csv("train_features.csv")
    
    # Prepare training data
    feature_cols = [col for col in train_features_df.columns if col not in ['window_id', 'label', 'event_sclk', 'split']]
    X_train = train_features_df[feature_cols].values
    y_train = train_features_df['label'].values
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Train Random Forest model
    print("Training Random Forest model...")
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
    
    # Filter out 'Omit' labels and convert to binary
    valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    X_val = valid_df[feature_cols].values
    y_val = valid_df['label'].values
    
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y_val)}")
    
    return rf_model, X_val, y_val, feature_cols

def find_optimal_thresholds(y_true, y_proba):
    """Find optimal thresholds using multiple criteria."""
    print("\n" + "="*70)
    print("FINDING OPTIMAL THRESHOLDS")
    print("="*70)
    
    # 1. ROC-based threshold (Youden's J statistic)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J = TPR - FPR
    youden_j = tpr - fpr
    optimal_roc_idx = np.argmax(youden_j)
    threshold_roc = roc_thresholds[optimal_roc_idx]
    
    print(f"\n1. ROC-Based Threshold (Youden's J):")
    print(f"   Threshold: {threshold_roc:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    
    # 2. F1-score optimization
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # Find threshold that maximizes F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_pr_idx = np.argmax(f1_scores)
    threshold_f1 = pr_thresholds[optimal_pr_idx]
    
    print(f"\n2. F1-Score Optimization:")
    print(f"   Threshold: {threshold_f1:.4f}")
    print(f"   PR AUC: {pr_auc:.4f}")
    
    # 3. Cost-sensitive threshold
    # Assume: FP cost = 1 (false alarm), FN cost = 10 (missed vortex)
    fp_cost, fn_cost = 1, 10
    
    def cost_function(threshold):
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fp * fp_cost + fn * fn_cost
        return total_cost
    
    # Test different thresholds to find minimum cost
    test_thresholds = np.linspace(0.1, 0.9, 81)
    costs = [cost_function(thresh) for thresh in test_thresholds]
    min_cost_idx = np.argmin(costs)
    threshold_cost = test_thresholds[min_cost_idx]
    min_cost = costs[min_cost_idx]
    
    print(f"\n3. Cost-Sensitive Threshold (FP cost=1, FN cost=10):")
    print(f"   Threshold: {threshold_cost:.4f}")
    print(f"   Expected cost: {min_cost:.2f}")
    
    # 4. Mission-specific threshold (high precision requirement)
    # Find threshold that gives at least 80% precision
    high_precision_indices = precision >= 0.8
    if np.any(high_precision_indices) and len(high_precision_indices) == len(pr_thresholds):
        valid_recall = recall[high_precision_indices]
        valid_thresholds = pr_thresholds[high_precision_indices]
        # Among high-precision thresholds, pick the one with highest recall
        if len(valid_recall) > 0:
            optimal_mission_idx = np.argmax(valid_recall)
            threshold_mission = valid_thresholds[optimal_mission_idx]
        else:
            threshold_mission = threshold_f1
    else:
        # Fallback: use F1-optimal threshold
        threshold_mission = threshold_f1
    
    print(f"\n4. Mission-Specific Threshold (min precision=80%):")
    print(f"   Threshold: {threshold_mission:.4f}")
    
    return {
        'roc_youden': threshold_roc,
        'f1_optimal': threshold_f1,
        'cost_sensitive': threshold_cost,
        'mission_specific': threshold_mission
    }

def evaluate_thresholds(y_true, y_proba, thresholds):
    """Evaluate all thresholds on validation set."""
    print(f"\n" + "="*70)
    print("EVALUATING THRESHOLDS ON VALIDATION SET")
    print("="*70)
    
    results = {}
    
    for name, threshold in thresholds.items():
        # Make predictions
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Calculate false positive rate per hour
        # Assuming ~10 samples per minute = 600 samples per hour
        samples_per_hour = 600
        fp_per_hour = (fp / len(y_true)) * samples_per_hour
        
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
    
    return results

def recommend_threshold(results):
    """Recommend best threshold based on Mars mission requirements."""
    print(f"\n" + "="*70)
    print("THRESHOLD RECOMMENDATION FOR MARS MISSION")
    print("="*70)
    
    # Mission requirements for Mars vortex detection:
    # 1. Minimize false positives (power waste on Mars)
    # 2. Maintain reasonable recall (don't miss vortices)
    # 3. High precision preferred (reliable detections)
    
    # Score each threshold based on mission requirements
    scores = {}
    for name, metrics in results.items():
        # Scoring criteria (higher is better):
        # - High precision (weight: 3)
        # - Reasonable recall > 30% (weight: 2) 
        # - Low false positives per hour < 15 (weight: 2)
        # - High F1-score (weight: 1)
        
        precision_score = metrics['precision'] * 3
        recall_score = max(0, metrics['recall'] - 0.3) * 2 if metrics['recall'] > 0.3 else -1
        fp_score = max(0, (15 - metrics['fp_per_hour']) / 15) * 2 if metrics['fp_per_hour'] < 15 else 0
        f1_score = metrics['f1_score'] * 1
        
        total_score = precision_score + recall_score + fp_score + f1_score
        scores[name] = total_score
    
    # Find best threshold
    best_threshold_name = max(scores.items(), key=lambda x: x[1])[0]
    best_threshold = results[best_threshold_name]
    
    print(f"Recommended threshold: {best_threshold_name.replace('_', ' ').title()}")
    print(f"Threshold value: {best_threshold['threshold']:.4f}")
    
    print(f"\nExpected performance on Mars:")
    print(f"  Precision: {best_threshold['precision']:.4f} ({best_threshold['precision']*100:.1f}%)")
    print(f"  Recall:    {best_threshold['recall']:.4f} ({best_threshold['recall']*100:.1f}%)")
    print(f"  F1-Score:  {best_threshold['f1_score']:.4f}")
    print(f"  False positives per hour: {best_threshold['fp_per_hour']:.1f}")
    
    print(f"\nMission impact:")
    print(f"  • {best_threshold['precision']*100:.1f}% of detections will be real vortices")
    print(f"  • {best_threshold['recall']*100:.1f}% of actual vortices will be detected")
    print(f"  • ~{best_threshold['fp_per_hour']:.0f} false alarms per hour")
    
    return best_threshold['threshold'], best_threshold_name

def plot_threshold_analysis(y_true, y_proba, thresholds, results):
    """Create visualization of threshold analysis."""
    print(f"\nGenerating threshold analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Threshold Calibration - Mars Vortex Detection', fontsize=16, fontweight='bold')
    
    # 1. ROC Curve with optimal thresholds
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    
    # Mark optimal thresholds
    for name, threshold in thresholds.items():
        # Find closest threshold in ROC curve
        idx = np.argmin(np.abs(roc_thresholds - threshold))
        axes[0, 0].plot(fpr[idx], tpr[idx], 'o', markersize=8,
                       label=f'{name.replace("_", " ").title()}')
    
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve with Optimal Thresholds')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
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
    test_thresholds = np.linspace(0.1, 0.9, 81)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in test_thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
    
    axes[1, 0].plot(test_thresholds, precisions, label='Precision', color='blue', linewidth=2)
    axes[1, 0].plot(test_thresholds, recalls, label='Recall', color='green', linewidth=2)
    axes[1, 0].plot(test_thresholds, f1_scores, label='F1-Score', color='red', linewidth=2)
    
    # Mark optimal thresholds
    for name, threshold in thresholds.items():
        axes[1, 0].axvline(x=threshold, linestyle='--', alpha=0.7, linewidth=2,
                          label=f'{name.replace("_", " ").title()}')
    
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Metrics vs Threshold')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance Comparison Bar Chart
    threshold_names = list(results.keys())
    precisions = [results[name]['precision'] for name in threshold_names]
    recalls = [results[name]['recall'] for name in threshold_names]
    f1_scores = [results[name]['f1_score'] for name in threshold_names]
    
    x = np.arange(len(threshold_names))
    width = 0.25
    
    axes[1, 1].bar(x - width, precisions, width, label='Precision', alpha=0.8, color='blue')
    axes[1, 1].bar(x, recalls, width, label='Recall', alpha=0.8, color='green')
    axes[1, 1].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='red')
    
    axes[1, 1].set_xlabel('Threshold Method')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Performance Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([name.replace('_', ' ').title() for name in threshold_names], 
                              rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mars_vortex_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: mars_vortex_threshold_analysis.png")
    plt.show()

def main():
    """Main threshold tuning pipeline."""
    print("="*70)
    print("THRESHOLD TUNING FOR MARS VORTEX DETECTION")
    print("="*70)
    print("Optimizing decision threshold for Mars rover deployment")
    print("="*70)
    
    # Load data and train model
    rf_model, X_val, y_val, feature_cols = load_data_and_model()
    
    # Get prediction probabilities
    y_proba = rf_model.predict_proba(X_val)[:, 1]
    
    # Find optimal thresholds
    thresholds = find_optimal_thresholds(y_val, y_proba)
    
    # Evaluate all thresholds
    results = evaluate_thresholds(y_val, y_proba, thresholds)
    
    # Get recommendation
    recommended_threshold, recommended_name = recommend_threshold(results)
    
    # Create visualizations
    plot_threshold_analysis(y_val, y_proba, thresholds, results)
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('threshold_calibration_results.csv')
    print(f"\n  Saved: threshold_calibration_results.csv")
    
    print(f"\n" + "="*70)
    print("THRESHOLD CALIBRATION COMPLETED")
    print("="*70)
    print(f"Recommended threshold: {recommended_threshold:.4f}")
    print(f"Method: {recommended_name.replace('_', ' ').title()}")
    print(f"Ready for Mars deployment!")
    
    return recommended_threshold, results_df

if __name__ == "__main__":
    recommended_threshold, results = main()
