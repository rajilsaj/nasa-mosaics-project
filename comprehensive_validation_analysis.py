#!/usr/bin/env python3
"""
Comprehensive Validation Set Analysis for Random Forest Classifier
===================================================================

This script performs a complete analysis of the Random Forest classifier
on the validation set, including:
- Performance metrics across multiple thresholds
- Precision-Recall and ROC curves
- Confusion matrix analysis
- Feature importance correlation with performance
- Comparison summary with test set results
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = "results"
FIGURES_DIR = "results"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_model_and_validation_data():
    """Load trained Random Forest model and validation data."""
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION SET ANALYSIS")
    print("=" * 70)
    
    # Find latest improved model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) 
                   if f.startswith("improved_rf_vortex_detector_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No improved model found. Run improved_train_rf_model.py first.")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    print(f"\nLoading model: {latest_model}")
    model = joblib.load(model_path)
    
    # Load validation sliding features
    val_features_df = pd.read_csv("datasets/val_sliding_features.csv")
    print(f"Loaded {len(val_features_df):,} validation feature vectors")
    
    # Prepare validation data
    valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    # Get feature columns
    train_df = pd.read_csv("datasets/train_features.csv")
    train_feature_cols = [col for col in train_df.columns 
                          if col not in ['window_id', 'label', 'event_sclk']]
    
    feature_cols = [col for col in train_feature_cols 
                   if col in valid_df.columns]
    
    X_val = valid_df[feature_cols].values
    y_val = valid_df['label'].values
    
    print(f"\nValidation Data Summary:")
    print(f"  Total samples: {len(X_val):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution: {np.bincount(y_val)}")
    print(f"  Class ratio: {np.bincount(y_val)[0] / np.bincount(y_val)[1]:.1f}:1 (Negative:Positive)")
    
    return model, X_val, y_val, feature_cols, valid_df

# =============================================================================
# THRESHOLD EVALUATION
# =============================================================================

def evaluate_multiple_thresholds(model, X_val, y_val, thresholds):
    """Evaluate Random Forest at multiple decision thresholds."""
    print("\n" + "=" * 70)
    print("EVALUATING MULTIPLE THRESHOLDS")
    print("=" * 70)
    
    # Get model probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    print(f"Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}")
    
    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'fpr': fpr,
            'fnr': fnr
        })
        
        print(f"\nThreshold {threshold:.3f}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  TP: {tp:4d}, FP: {fp:4d}, TN: {tn:6d}, FN: {fn:4d}")
        print(f"  FPR: {fpr:.4f}, FNR: {fnr:.4f}")
    
    return results, y_proba

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_validation_plots(results, y_val, y_proba, output_dir, model, feature_cols):
    """Create comprehensive validation analysis plots."""
    print("\n" + "=" * 70)
    print("CREATING VALIDATION ANALYSIS PLOTS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Precision-Recall Curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val, y_proba)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(recall_curve, precision_curve, 'b-', linewidth=2, label='PR Curve')
    
    # Mark evaluated thresholds
    thresholds = [r['threshold'] for r in results]
    for result in results:
        ax1.scatter(result['recall'], result['precision'], 
                   s=100, zorder=5, alpha=0.7)
        ax1.annotate(f"{result['threshold']:.2f}", 
                    (result['recall'], result['precision']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. ROC Curve
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_val, y_proba)
    roc_auc = roc_auc_score(y_val, y_proba)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fpr_curve, tpr_curve, 'r-', linewidth=2, 
            label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Metrics vs Threshold
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1_score'] for r in results]
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(thresholds, precisions, 'r-o', label='Precision', linewidth=2, markersize=8)
    ax3.plot(thresholds, recalls, 'b-s', label='Recall', linewidth=2, markersize=8)
    ax3.plot(thresholds, f1_scores, 'g-^', label='F1-Score', linewidth=2, markersize=8)
    ax3.set_xlabel('Decision Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Confusion Matrix Heatmap (best F1 threshold)
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    best_result = results[best_f1_idx]
    cm = np.array([[best_result['tn'], best_result['fp']],
                   [best_result['fn'], best_result['tp']]])
    
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    ax4.set_title(f'Confusion Matrix (Threshold={best_result["threshold"]:.3f}, F1={best_result["f1_score"]:.3f})',
                 fontsize=12, fontweight='bold')
    
    # 5. Error Rates
    fprs = [r['fpr'] for r in results]
    fnrs = [r['fnr'] for r in results]
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(thresholds, fprs, 'r-o', label='False Positive Rate', linewidth=2, markersize=8)
    ax5.plot(thresholds, fnrs, 'b-s', label='False Negative Rate', linewidth=2, markersize=8)
    ax5.set_xlabel('Decision Threshold', fontsize=12)
    ax5.set_ylabel('Rate', fontsize=12)
    ax5.set_title('Error Rates vs Threshold', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Performance Summary Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_data = []
    for r in results:
        summary_data.append([
            f"{r['threshold']:.3f}",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1_score']:.3f}",
            f"{r['accuracy']:.3f}"
        ])
    
    table = ax6.table(cellText=summary_data,
                      colLabels=['Threshold', 'Precision', 'Recall', 'F1', 'Accuracy'],
                      cellLoc='center',
                      loc='center',
                      bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax6.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # 7. Probability Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(y_proba[y_val == 0], bins=50, alpha=0.6, label='Negative Class', color='blue', density=True)
    ax7.hist(y_proba[y_val == 1], bins=50, alpha=0.6, label='Positive Class', color='red', density=True)
    ax7.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    ax7.set_xlabel('Predicted Probability', fontsize=12)
    ax7.set_ylabel('Density', fontsize=12)
    ax7.set_title('Probability Distribution by Class', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Feature Importance (if available)
    ax8 = fig.add_subplot(gs[2, 1])
    feature_importance = getattr(model, "feature_importances_", None)
    if feature_importance is not None and len(feature_importance) == len(feature_cols):
        top_indices = np.argsort(feature_importance)[-10:][::-1]
        top_features = [feature_cols[i] for i in top_indices]
        top_importance = feature_importance[top_indices]

        ax8.barh(range(len(top_features)), top_importance, color='steelblue')
        ax8.set_yticks(range(len(top_features)))
        ax8.set_yticklabels(top_features)
        ax8.set_xlabel('Importance', fontsize=12)
        ax8.set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
        ax8.invert_yaxis()
        ax8.grid(True, alpha=0.3, axis='x')
    else:
        ax8.text(0.5, 0.5, 'Feature importance\nnot available',
                 ha='center', va='center', fontsize=12)
        ax8.axis('off')
    
    # 9. Metrics Comparison Bar Chart
    ax9 = fig.add_subplot(gs[2, 2])
    metrics_data = {
        'Precision': [r['precision'] for r in results],
        'Recall': [r['recall'] for r in results],
        'F1-Score': [r['f1_score'] for r in results]
    }
    x = np.arange(len(thresholds))
    width = 0.25
    
    for i, (metric, values) in enumerate(metrics_data.items()):
        ax9.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax9.set_xlabel('Threshold Index', fontsize=12)
    ax9.set_ylabel('Score', fontsize=12)
    ax9.set_title('Metrics Comparison', fontsize=14, fontweight='bold')
    ax9.set_xticks(x + width)
    ax9.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"comprehensive_validation_analysis_{timestamp}.png")
    plt.suptitle('Random Forest Classifier - Comprehensive Validation Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved comprehensive plot: {filename}")
    plt.close()
    
    return filename

# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(results, y_val, y_proba, output_dir):
    """Generate a comprehensive text summary report."""
    print("\n" + "=" * 70)
    print("GENERATING SUMMARY REPORT")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"validation_analysis_report_{timestamp}.txt")
    
    roc_auc = roc_auc_score(y_val, y_proba)
    
    # Find best thresholds
    best_f1_idx = max(range(len(results)), key=lambda i: results[i]['f1_score'])
    best_recall_idx = max(range(len(results)), key=lambda i: results[i]['recall'])
    best_precision_idx = max(range(len(results)), key=lambda i: results[i]['precision'])
    
    with open(report_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RANDOM FOREST CLASSIFIER - VALIDATION SET ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total validation samples: {len(y_val):,}\n")
        f.write(f"Positive samples: {np.sum(y_val):,} ({np.mean(y_val)*100:.2f}%)\n")
        f.write(f"Negative samples: {np.sum(1-y_val):,} ({np.mean(1-y_val)*100:.2f}%)\n")
        f.write(f"Class ratio: {np.sum(1-y_val) / np.sum(y_val):.1f}:1\n\n")
        
        f.write(f"MODEL DISCRIMINATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}\n\n")
        
        f.write("THRESHOLD EVALUATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['threshold']:<12.3f} {r['precision']:<12.4f} {r['recall']:<12.4f} "
                    f"{r['f1_score']:<12.4f} {r['accuracy']:<12.4f}\n")
        f.write("\n")
        
        f.write("BEST PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        
        best_f1 = results[best_f1_idx]
        f.write(f"\n1. BEST F1-SCORE (Threshold = {best_f1['threshold']:.3f}):\n")
        f.write(f"   Precision: {best_f1['precision']:.4f}\n")
        f.write(f"   Recall:    {best_f1['recall']:.4f}\n")
        f.write(f"   F1-Score:  {best_f1['f1_score']:.4f}\n")
        f.write(f"   Accuracy:  {best_f1['accuracy']:.4f}\n")
        f.write(f"   TP: {best_f1['tp']}, FP: {best_f1['fp']}, TN: {best_f1['tn']}, FN: {best_f1['fn']}\n")
        
        best_recall = results[best_recall_idx]
        f.write(f"\n2. HIGHEST RECALL (Threshold = {best_recall['threshold']:.3f}):\n")
        f.write(f"   Precision: {best_recall['precision']:.4f}\n")
        f.write(f"   Recall:    {best_recall['recall']:.4f}\n")
        f.write(f"   F1-Score:  {best_recall['f1_score']:.4f}\n")
        
        best_precision = results[best_precision_idx]
        f.write(f"\n3. HIGHEST PRECISION (Threshold = {best_precision['threshold']:.3f}):\n")
        f.write(f"   Precision: {best_precision['precision']:.4f}\n")
        f.write(f"   Recall:    {best_precision['recall']:.4f}\n")
        f.write(f"   F1-Score:  {best_precision['f1_score']:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"1. The Random Forest achieves ROC AUC of {roc_auc:.3f}, indicating ")
        f.write(f"{'good' if roc_auc > 0.7 else 'moderate'} discrimination capability.\n")
        f.write(f"2. Best F1-score of {best_f1['f1_score']:.3f} at threshold {best_f1['threshold']:.3f}.\n")
        f.write(f"3. Precision ranges from {min(r['precision'] for r in results):.3f} to ")
        f.write(f"{max(r['precision'] for r in results):.3f} across thresholds.\n")
        f.write(f"4. Recall ranges from {min(r['recall'] for r in results):.3f} to ")
        f.write(f"{max(r['recall'] for r in results):.3f} across thresholds.\n")
    
    print(f"Saved report: {report_file}")
    return report_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    # Load data and model
    model, X_val, y_val, feature_cols, valid_df = load_model_and_validation_data()
    
    # Evaluate at multiple thresholds
    thresholds = [0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    results, y_proba = evaluate_multiple_thresholds(model, X_val, y_val, thresholds)
    
    # Create visualizations
    plot_file = create_validation_plots(results, y_val, y_proba, FIGURES_DIR, model, feature_cols)
    
    # Generate summary report
    report_file = generate_summary_report(results, y_val, y_proba, OUTPUT_DIR)
    
    # Save results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = os.path.join(OUTPUT_DIR, f"validation_analysis_results_{timestamp}.json")
    
    # Convert numpy types to native Python types for JSON serialization
    json_results = []
    for r in results:
        json_results.append({
            'threshold': float(r['threshold']),
            'accuracy': float(r['accuracy']),
            'precision': float(r['precision']),
            'recall': float(r['recall']),
            'f1_score': float(r['f1_score']),
            'tp': int(r['tp']),
            'fp': int(r['fp']),
            'tn': int(r['tn']),
            'fn': int(r['fn']),
            'fpr': float(r['fpr']),
            'fnr': float(r['fnr'])
        })
    
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_samples': int(len(y_val)),
            'class_distribution': {
                'negative': int(np.sum(1-y_val)), 
                'positive': int(np.sum(y_val))
            },
            'roc_auc': float(roc_auc_score(y_val, y_proba)),
            'results': json_results
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("VALIDATION ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Plot saved: {plot_file}")
    print(f"Report saved: {report_file}")
    print(f"Results saved: {json_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

