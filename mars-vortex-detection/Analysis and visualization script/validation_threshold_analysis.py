#!/usr/bin/env python3
"""
Validation Threshold Analysis with Plotting
Creates threshold comparison plots and results for validation set
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

def load_model_and_validation_data():
    """Load trained model and validation data."""
    print("=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS WITH PLOTTING")
    print("=" * 70)
    
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
    
    # Load validation features
    val_features_df = pd.read_csv("val_sliding_features.csv")
    print(f"Loaded {len(val_features_df)} validation feature vectors")
    
    # Prepare validation data
    valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    # Get feature columns, excluding metadata columns
    feature_cols = [col for col in valid_df.columns if col not in ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label']]
    
    # Load training features to get the exact feature order
    train_df = pd.read_csv("train_features.csv")
    train_feature_cols = [col for col in train_df.columns if col not in ['window_id', 'label', 'event_sclk']]
    
    # Ensure validation features match training features
    feature_cols = [col for col in train_feature_cols if col in feature_cols]
    
    print(f"Training features: {len(train_feature_cols)} - {train_feature_cols}")
    print(f"Validation features: {len(feature_cols)} - {feature_cols}")
    
    X_val = valid_df[feature_cols].values
    y_val = valid_df['label'].values
    
    print(f"Validation samples: {len(X_val)}")
    print(f"Class distribution: {np.bincount(y_val)}")
    print(f"Class ratio: {np.bincount(y_val)[0] / np.bincount(y_val)[1]:.1f}:1 (Negative:Positive)")
    
    return model, X_val, y_val, feature_cols, valid_df

def evaluate_validation_thresholds(model, X_val, y_val, thresholds):
    """Evaluate multiple thresholds on validation set."""
    print(f"\n" + "=" * 70)
    print("VALIDATION THRESHOLD EVALUATION")
    print("=" * 70)
    
    # Get model probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    print(f"Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}")
    
    results = []
    for threshold in thresholds:
        # Apply threshold
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = (y_pred == y_val).mean()
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        result = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'fpr': fpr,
            'fnr': fnr
        }
        results.append(result)
        
        print(f"\nThreshold = {threshold:.3f}:")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  FPR: {fpr:.3f} ({fpr*100:.1f}%), FNR: {fnr:.3f} ({fnr*100:.1f}%)")
    
    return results, y_proba

def plot_validation_threshold_analysis(results, y_val, y_proba, output_dir="results"):
    """Plot comprehensive threshold analysis for validation set."""
    print(f"\nCreating validation threshold analysis plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate precision-recall curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val, y_proba)
    f1_curve = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    
    # Calculate ROC curve
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_val, y_proba)
    
    # Extract metrics from results
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Validation Set Threshold Analysis - Extreme Class Imbalance', fontsize=16, fontweight='bold')
    
    # 1. Precision-Recall Curve
    axes[0, 0].plot(recall_curve, precision_curve, 'b-', linewidth=2, label='PR Curve')
    axes[0, 0].scatter(recalls, precisions, c='red', s=50, zorder=5, label='Tested Thresholds')
    for i, threshold in enumerate(thresholds):
        axes[0, 0].annotate(f'{threshold:.2f}', (recalls[i], precisions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. F1-Score vs Threshold
    axes[0, 1].plot(pr_thresholds, f1_curve[:-1], 'g-', linewidth=2, label='F1-Score Curve', alpha=0.7)
    axes[0, 1].scatter(thresholds, f1_scores, c='red', s=50, zorder=5, label='Tested Thresholds')
    for i, threshold in enumerate(thresholds):
        axes[0, 1].annotate(f'{threshold:.2f}', (threshold, f1_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Precision vs Threshold
    axes[0, 2].plot(pr_thresholds, precision_curve[:-1], 'r-', linewidth=2, label='Precision Curve', alpha=0.7)
    axes[0, 2].scatter(thresholds, precisions, c='red', s=50, zorder=5, label='Tested Thresholds')
    for i, threshold in enumerate(thresholds):
        axes[0, 2].annotate(f'{threshold:.2f}', (threshold, precisions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 2].set_xlabel('Threshold')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Precision vs Threshold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # 4. Recall vs Threshold
    axes[1, 0].plot(pr_thresholds, recall_curve[:-1], 'm-', linewidth=2, label='Recall Curve', alpha=0.7)
    axes[1, 0].scatter(thresholds, recalls, c='red', s=50, zorder=5, label='Tested Thresholds')
    for i, threshold in enumerate(thresholds):
        axes[1, 0].annotate(f'{threshold:.2f}', (threshold, recalls[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Recall vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 5. ROC Curve
    axes[1, 1].plot(fpr_curve, tpr_curve, 'purple', linewidth=2, label=f'ROC Curve (AUC = {roc_auc_score(y_val, y_proba):.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # 6. Threshold Comparison Bar Chart
    x_pos = np.arange(len(thresholds))
    width = 0.25
    
    axes[1, 2].bar(x_pos - width, precisions, width, label='Precision', alpha=0.8)
    axes[1, 2].bar(x_pos, recalls, width, label='Recall', alpha=0.8)
    axes[1, 2].bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    axes[1, 2].set_xlabel('Threshold')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Metrics Comparison by Threshold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels([f'{t:.2f}' for t in thresholds])
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"validation_threshold_analysis_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation threshold analysis plot saved to: {plot_filename}")
    return plot_filename

def print_validation_threshold_summary(results):
    """Print detailed validation threshold results."""
    print(f"\n" + "=" * 70)
    print("VALIDATION THRESHOLD COMPARISON SUMMARY")
    print("=" * 70)
    
    # Create summary table
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['threshold']:<10.3f} {result['precision']:<10.3f} {result['recall']:<10.3f} "
              f"{result['f1_score']:<10.3f} {result['tp']:<6d} {result['fp']:<6d} "
              f"{result['tn']:<6d} {result['fn']:<6d}")
    
    print(f"\n" + "=" * 70)
    print("MISSION-SPECIFIC RECOMMENDATIONS")
    print("=" * 70)
    
    # Find best thresholds for different objectives
    max_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    max_recall_idx = max(range(len(results)), key=lambda i: results[i]['recall'])
    max_precision_idx = max(range(len(results)), key=lambda i: results[i]['precision'])
    
    print(f"\n1. MAXIMUM F1-SCORE:")
    best_f1 = results[max_f1_idx]
    print(f"   Threshold: {best_f1['threshold']:.3f}")
    print(f"   Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}, F1: {best_f1['f1_score']:.3f}")
    print(f"   Use Case: General purpose, balanced precision/recall")
    
    print(f"\n2. HIGHEST RECALL:")
    best_recall = results[max_recall_idx]
    print(f"   Threshold: {best_recall['threshold']:.3f}")
    print(f"   Precision: {best_recall['precision']:.3f}, Recall: {best_recall['recall']:.3f}, F1: {best_recall['f1_score']:.3f}")
    print(f"   Use Case: Science priority, don't miss vortices")
    
    print(f"\n3. HIGHEST PRECISION:")
    best_precision = results[max_precision_idx]
    print(f"   Threshold: {best_precision['threshold']:.3f}")
    print(f"   Precision: {best_precision['precision']:.3f}, Recall: {best_precision['recall']:.3f}, F1: {best_precision['f1_score']:.3f}")
    print(f"   Use Case: Energy conservation, minimize false alarms")

def save_validation_results(results, output_dir="results"):
    """Save validation threshold results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to JSON-serializable format
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if key == 'confusion_matrix':
                json_result[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                json_result[key] = int(value)
            elif isinstance(value, (np.float64, np.float32, np.float16)):
                json_result[key] = float(value)
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    # Save results
    results_filename = os.path.join(output_dir, f"validation_threshold_results_{timestamp}.json")
    with open(results_filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nValidation threshold results saved to: {results_filename}")
    return results_filename

def main():
    """Main execution function."""
    print("Starting validation threshold analysis with plotting...")
    
    # Step 1: Load model and validation data
    model, X_val, y_val, feature_cols, valid_df = load_model_and_validation_data()
    
    # Step 2: Test multiple thresholds
    thresholds = [0.450, 0.600, 0.750, 0.900]  # Same as test set
    results, y_proba = evaluate_validation_thresholds(model, X_val, y_val, thresholds)
    
    # Step 3: Create comprehensive plots
    plot_filename = plot_validation_threshold_analysis(results, y_val, y_proba)
    
    # Step 4: Print detailed summary
    print_validation_threshold_summary(results)
    
    # Step 5: Save results
    results_filename = save_validation_results(results)
    
    print(f"\n" + "=" * 70)
    print("VALIDATION THRESHOLD ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Plot saved to: {plot_filename}")
    print(f"Results saved to: {results_filename}")
    print("=" * 70)

if __name__ == "__main__":
    main()
