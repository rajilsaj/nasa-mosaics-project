#!/usr/bin/env python3
"""
Improved Test Evaluation with Optimized Threshold
Tests the model on sliding window test data with threshold from validation tuning
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from datetime import datetime

def load_model_and_test_data():
    """Load trained model and test data."""
    print("=" * 70)
    print("IMPROVED TEST EVALUATION WITH OPTIMIZED THRESHOLD")
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
    
    # Load test features
    test_features_df = pd.read_csv("datasets/test_sliding_features.csv")
    print(f"Loaded {len(test_features_df)} test feature vectors")
    
    # Prepare test data
    valid_df = test_features_df[test_features_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    # Get feature columns, excluding metadata columns
    feature_cols = [col for col in valid_df.columns if col not in ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label']]
    
    # Load training features to get the exact feature order
    train_df = pd.read_csv("datasets/train_features.csv")
    train_feature_cols = [col for col in train_df.columns if col not in ['window_id', 'label', 'event_sclk']]
    
    # Ensure test features match training features
    feature_cols = [col for col in train_feature_cols if col in feature_cols]
    
    print(f"Training features: {len(train_feature_cols)} - {train_feature_cols}")
    print(f"Test features: {len(feature_cols)} - {feature_cols}")
    
    X_test = valid_df[feature_cols].values
    y_test = valid_df['label'].values
    
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution: {np.bincount(y_test)}")
    print(f"Class ratio: {np.bincount(y_test)[0] / np.bincount(y_test)[1]:.1f}:1 (Negative:Positive)")
    
    return model, X_test, y_test, feature_cols, valid_df

def evaluate_with_threshold(model, X_test, y_test, threshold):
    """Evaluate model with specific threshold."""
    print(f"\n" + "=" * 70)
    print(f"TEST EVALUATION WITH THRESHOLD = {threshold:.3f}")
    print("=" * 70)
    
    # Get model probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}")
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {tn:8d}  {fp:8d}")
    print(f"Positive  {fn:8d}  {tp:8d}")
    
    # Error analysis
    print(f"\nError Analysis:")
    print(f"  True Positives:  {tp:4d} (correctly detected vortices)")
    print(f"  False Positives: {fp:4d} (incorrectly flagged as vortex)")
    print(f"  True Negatives:  {tn:4d} (correctly identified non-vortex)")
    print(f"  False Negatives: {fn:4d} (missed vortex detections)")
    
    # Calculate rates
    if fp + tn > 0:
        fpr = fp / (fp + tn)
        print(f"  False Positive Rate: {fpr:.4f} ({fpr*100:.2f}%)")
    
    if tp + fn > 0:
        fnr = fn / (tp + fn)
        print(f"  False Negative Rate: {fnr:.4f} ({fnr*100:.2f}%)")
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'probabilities': y_proba,
        'predictions': y_pred
    }

def compare_thresholds(model, X_test, y_test, thresholds):
    """Compare multiple thresholds."""
    print(f"\n" + "=" * 70)
    print("THRESHOLD COMPARISON ON TEST SET")
    print("=" * 70)
    
    results = []
    for threshold in thresholds:
        result = evaluate_with_threshold(model, X_test, y_test, threshold)
        results.append(result)
    
    # Create comparison table
    print(f"\n" + "=" * 70)
    print("THRESHOLD COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC AUC':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['threshold']:<10.3f} {result['precision']:<10.3f} {result['recall']:<10.3f} {result['f1_score']:<10.3f} {result['roc_auc']:<10.3f}")
    
    return results

def save_test_results(results, output_dir="results"):
    """Save test evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_filename = os.path.join(output_dir, f"test_evaluation_{timestamp}.json")
    with open(results_filename, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            if 'confusion_matrix' in json_result:
                json_result['confusion_matrix'] = json_result['confusion_matrix'].tolist()
            if 'probabilities' in json_result:
                json_result['probabilities'] = json_result['probabilities'].tolist()
            if 'predictions' in json_result:
                json_result['predictions'] = json_result['predictions'].tolist()
            json_results.append(json_result)
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nTest evaluation results saved to: {results_filename}")
    return results_filename

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Evaluate improved model on test set with optimized threshold')
    parser.add_argument('--threshold', type=float, default=0.450, 
                       help='Threshold to use for evaluation (default: 0.450 for high recall)')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare multiple thresholds')
    args = parser.parse_args()
    
    print("Starting improved test evaluation...")
    
    # Step 1: Load model and test data
    model, X_test, y_test, feature_cols, valid_df = load_model_and_test_data()
    
    if args.compare:
        # Compare multiple thresholds
        thresholds = [0.450, 0.600, 0.750, 0.900]  # High recall to high precision
        results = compare_thresholds(model, X_test, y_test, thresholds)
    else:
        # Single threshold evaluation
        result = evaluate_with_threshold(model, X_test, y_test, args.threshold)
        results = [result]
    
    # Save results
    results_filename = save_test_results(results)
    
    print(f"\n" + "=" * 70)
    print("TEST EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Results saved to: {results_filename}")
    print(f"Model performance on real-world sliding window test data")
    print("=" * 70)

if __name__ == "__main__":
    main()



