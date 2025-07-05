# evaluation_utils.py
"""
Evaluation utilities for LSTM vortex detection.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(y_true, y_pred):
    """
    Computes and prints precision, recall, F1-score, and confusion matrix.
    Returns precision, recall, f1, (tn, fp, fn, tp)
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    return precision, recall, f1, (tn, fp, fn, tp)

def sweep_confidence_thresholds(y_true, y_pred, y_pred_proba, thresholds):
    """
    Sweeps confidence thresholds, applies to y_pred, computes and prints metrics for each.
    Returns best F1 and its threshold.
    """
    best_f1 = 0
    best_thresh = 0
    for thresh in thresholds:
        y_pred_thresh = y_pred.copy()
        for i, pred in enumerate(y_pred_thresh):
            if pred == 1 and y_pred_proba[i] < thresh:
                y_pred_thresh[i] = 0
        precision = precision_score(y_true, y_pred_thresh, zero_division=0)
        recall = recall_score(y_true, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
        print(f"Thresh: {thresh:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"\nBest F1-Score (After Combined Filters): {best_f1:.4f} at confidence threshold {best_thresh:.2f}")
    return best_f1, best_thresh 