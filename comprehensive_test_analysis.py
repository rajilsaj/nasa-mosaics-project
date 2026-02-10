#!/usr/bin/env python3
"""
Comprehensive Test Set Analysis
================================
Generate full performance diagnostics for the Random Forest classifier on the
natural test distribution (sliding-window test set). Includes threshold sweep,
PR/ROC curves, confusion matrix, error rate plots, probability histograms, and
feature importance.
"""

import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

MODEL_PATH = "models/improved_rf_vortex_detector_20251010_114031.pkl"
TEST_FEATURES_FILE = "test_sliding_features.csv"
OUTPUT_DIR = "results"
FIGURES_DIR = "results"
DEFAULT_THRESHOLDS = [0.45, 0.60, 0.75, 0.90]


def load_model_and_test_data():
    model = joblib.load(MODEL_PATH)
    test_features_df = pd.read_csv(TEST_FEATURES_FILE)
    valid_df = test_features_df[test_features_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    feature_cols = [col for col in valid_df.columns
                    if col not in ['window_id', 'start_idx', 'end_idx',
                                   'start_sclk', 'end_sclk', 'label']]
    return model, valid_df, feature_cols


def evaluate_thresholds(model, test_df, feature_cols, thresholds):
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    y_proba = model.predict_proba(X_test)[:, 1]

    results = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'fpr': fpr,
            'fnr': fnr
        })
    return results, y_test, y_proba


def plot_test_analysis(results, y_test, y_proba, feature_cols, model):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Precision-Recall Curve
    ax1 = fig.add_subplot(gs[0, 0])
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    ax1.plot(recall_curve, precision_curve, 'b-', linewidth=2, label='PR Curve')
    for r in results:
        ax1.scatter(r['recall'], r['precision'], color='red', s=50)
        ax1.annotate(f"{r['threshold']:.2f}", (r['recall'], r['precision']),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.grid(True, alpha=0.3)

    # ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    ax2.plot(fpr_curve, tpr_curve, 'r-', linewidth=2, label=f'ROC (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Metrics vs Threshold
    ax3 = fig.add_subplot(gs[0, 2])
    thresholds = [r['threshold'] for r in results]
    ax3.plot(thresholds, [r['precision'] for r in results], 'r-o', label='Precision')
    ax3.plot(thresholds, [r['recall'] for r in results], 'b-s', label='Recall')
    ax3.plot(thresholds, [r['f1_score'] for r in results], 'g-^', label='F1-Score')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('Metrics vs Threshold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Confusion Matrix at best F1
    ax4 = fig.add_subplot(gs[1, 0])
    best_f1 = max(results, key=lambda r: r['f1_score'])
    cm = np.array([[best_f1['tn'], best_f1['fp']], [best_f1['fn'], best_f1['tp']]])
    cax = ax4.imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        ax4.text(j, i, f"{val}", ha='center', va='center', fontsize=12)
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Predicted Negative', 'Predicted Positive'], rotation=15)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Actual Negative', 'Actual Positive'])
    ax4.set_title(f"Confusion Matrix (Threshold={best_f1['threshold']:.2f}, F1={best_f1['f1_score']:.3f})")
    fig.colorbar(cax, ax=ax4, fraction=0.046, pad=0.04)

    # Error Rates vs Threshold
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(thresholds, [r['fpr'] for r in results], 'r-o', label='False Positive Rate')
    ax5.plot(thresholds, [r['fnr'] for r in results], 'b-s', label='False Negative Rate')
    ax5.set_xlabel('Threshold')
    ax5.set_ylabel('Rate')
    ax5.set_title('Error Rates vs Threshold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Performance table
    ax6 = fig.add_subplot(gs[1, 2])
    table_data = []
    for r in results:
        table_data.append([f"{r['threshold']:.2f}", f"{r['precision']:.3f}",
                           f"{r['recall']:.3f}", f"{r['f1_score']:.3f}",
                           f"{r['accuracy']:.3f}"])
    table = ax6.table(cellText=table_data,
                      colLabels=['Threshold', 'Precision', 'Recall', 'F1', 'Accuracy'],
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    ax6.axis('off')
    ax6.set_title('Performance Summary', fontsize=14, fontweight='bold')

    # Probability distributions
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(y_proba[y_test == 0], bins=50, alpha=0.6, density=True, color='blue', label='Negative Class')
    ax7.hist(y_proba[y_test == 1], bins=50, alpha=0.6, density=True, color='red', label='Positive Class')
    ax7.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
    ax7.set_xlabel('Predicted Probability')
    ax7.set_ylabel('Density')
    ax7.set_title('Probability Distribution by Class')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Feature importance
    ax8 = fig.add_subplot(gs[2, 1])
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-10:][::-1]
        top_features = [feature_cols[i] for i in top_idx]
        top_values = importances[top_idx]
        ax8.barh(range(len(top_features)), top_values, color='steelblue')
        ax8.set_yticks(range(len(top_features)))
        ax8.set_yticklabels(top_features)
        ax8.invert_yaxis()
        ax8.set_xlabel('Importance')
        ax8.set_title('Top 10 Feature Importance')
        ax8.grid(True, axis='x', alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'Feature Importance\nNot Available', ha='center', va='center', fontsize=12)
        ax8.axis('off')

    # Metrics comparison bar chart
    ax9 = fig.add_subplot(gs[2, 2])
    width = 0.25
    x = np.arange(len(thresholds))
    ax9.bar(x - width, [r['precision'] for r in results], width, label='Precision', alpha=0.8)
    ax9.bar(x, [r['recall'] for r in results], width, label='Recall', alpha=0.8)
    ax9.bar(x + width, [r['f1_score'] for r in results], width, label='F1-Score', alpha=0.8)
    ax9.set_xticks(x)
    ax9.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax9.set_xlabel('Threshold')
    ax9.set_ylabel('Score')
    ax9.set_title('Metrics Comparison by Threshold')
    ax9.grid(True, axis='y', alpha=0.3)
    ax9.legend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = os.path.join(FIGURES_DIR, f"comprehensive_test_analysis_{timestamp}.png")
    plt.suptitle('Random Forest Classifier - Comprehensive Test Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive plot: {figure_path}")

    return figure_path


def generate_report(results, y_test, y_proba):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"test_analysis_report_{timestamp}.txt")
    roc_auc = roc_auc_score(y_test, y_proba)
    best_f1 = max(results, key=lambda r: r['f1_score'])
    best_recall = max(results, key=lambda r: r['recall'])
    best_precision = max(results, key=lambda r: r['precision'])

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RANDOM FOREST CLASSIFIER - TEST SET ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total test samples: {len(y_test):,}\n")
        f.write(f"Positive samples: {np.sum(y_test):,} ({np.mean(y_test) * 100:.2f}%)\n")
        f.write(f"Negative samples: {np.sum(1 - y_test):,} ({np.mean(1 - y_test) * 100:.2f}%)\n")
        f.write(f"Class ratio: {np.sum(1 - y_test) / np.sum(y_test):.1f}:1\n\n")

        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}\n\n")

        f.write("THRESHOLD EVALUATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Accuracy':<12}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['threshold']:<12.3f}{r['precision']:<12.4f}{r['recall']:<12.4f}{r['f1_score']:<12.4f}{r['accuracy']:<12.4f}\n")
        f.write("\n")

        def summarize(label, info):
            f.write(f"\n{label}\n")
            f.write("-" * 70 + "\n")
            f.write(f"Threshold: {info['threshold']:.3f}\n")
            f.write(f"Precision: {info['precision']:.4f}\n")
            f.write(f"Recall:    {info['recall']:.4f}\n")
            f.write(f"F1-Score:  {info['f1_score']:.4f}\n")
            f.write(f"Accuracy:  {info['accuracy']:.4f}\n")
            f.write(f"TP: {info['tp']}, FP: {info['fp']}, TN: {info['tn']}, FN: {info['fn']}\n")

        summarize('Best F1-Score', best_f1)
        summarize('Highest Recall', best_recall)
        summarize('Highest Precision', best_precision)

    print(f"Saved test summary report: {report_path}")
    return report_path


def main():
    print("Starting comprehensive test analysis...")
    model, test_df, feature_cols = load_model_and_test_data()
    print(f"Loaded {len(test_df):,} test samples (positives: {test_df['label'].sum()}, negatives: {len(test_df) - test_df['label'].sum()})")

    results, y_test, y_proba = evaluate_thresholds(model, test_df, feature_cols, DEFAULT_THRESHOLDS)
    figure_path = plot_test_analysis(results, y_test, y_proba, feature_cols, model)
    report_path = generate_report(results, y_test, y_proba)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"test_analysis_results_{timestamp}.json")
    pd.DataFrame(results).to_json(json_path, orient='records', indent=2)
    print(f"Saved detailed results: {json_path}")

    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Figure: {figure_path}")
    print(f"Report: {report_path}")
    print(f"Results JSON: {json_path}")


if __name__ == "__main__":
    main()




