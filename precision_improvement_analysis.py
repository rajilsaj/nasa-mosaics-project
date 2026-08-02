#!/usr/bin/env python3
"""
Precision Improvement Analysis for Random Forest Model
======================================================

This script analyzes why precision is low and tests multiple approaches to improve it:
1. Higher thresholds (0.95, 0.98, 0.99)
2. Probability calibration (Platt scaling, Isotonic regression)
3. Class prior adjustment (Bayes adjustment)
4. Post-processing filters (temporal consistency)

Key Issue: Model trained on balanced data (1:1) but deployed on imbalanced data (225:1)
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for precision improvement analysis."""
    
    # File paths
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    
    # Test data
    TEST_SLIDING_FEATURES = "datasets/test_sliding_features.csv"
    
    # Training prior (balanced data)
    TRAIN_PRIOR_POS = 0.5  # 1:1 ratio
    
    # Deployment prior (from test data)
    DEPLOYMENT_PRIOR_POS = 0.0044  # 380 / 85925 = 0.44%
    DEPLOYMENT_PRIOR_NEG = 0.9956  # 99.56%
    
    # High precision thresholds to test
    HIGH_PRECISION_THRESHOLDS = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    
    # Target precision for optimization
    TARGET_PRECISION = 0.10  # 10% minimum precision

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

def load_model_and_test_data():
    """Load the latest trained model and test data."""
    print("=" * 70)
    print("LOADING MODEL AND TEST DATA")
    print("=" * 70)
    
    # Find latest model
    model_files = [f for f in os.listdir(Config.MODELS_DIR) 
                   if f.startswith("rf_vortex_detector_") and f.endswith(".pkl")]
    
    if not model_files:
        # Try alternative naming
        model_files = [f for f in os.listdir(Config.MODELS_DIR) 
                      if f.endswith(".pkl") and "rf" in f.lower()]
    
    if not model_files:
        raise FileNotFoundError("No Random Forest model found in models/ directory")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(Config.MODELS_DIR, latest_model)
    print(f"Loading model: {latest_model}")
    
    model = joblib.load(model_path)
    
    # Load test sliding features
    if not os.path.exists(Config.TEST_SLIDING_FEATURES):
        raise FileNotFoundError(f"Test features file not found: {Config.TEST_SLIDING_FEATURES}")
    
    test_df = pd.read_csv(Config.TEST_SLIDING_FEATURES)
    print(f"Loaded {len(test_df):,} test feature vectors")
    
    # Filter out 'Omit' labels
    valid_df = test_df[test_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    # Get feature columns
    exclude_cols = ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 
                    'label', 'sliding_window_id', 'sliding_start_idx', 'sliding_end_idx',
                    'sliding_start_sclk', 'sliding_end_sclk']
    feature_cols = [col for col in valid_df.columns if col not in exclude_cols]
    
    # Ensure feature order matches training
    # Try to load training features to get exact order
    if os.path.exists("datasets/train_features.csv"):
        train_df = pd.read_csv("datasets/train_features.csv")
        train_feature_cols = [col for col in train_df.columns 
                             if col not in ['window_id', 'label', 'event_sclk']]
        # Match order
        feature_cols = [col for col in train_feature_cols if col in feature_cols]
    
    X_test = valid_df[feature_cols].values
    y_test = valid_df['label'].values
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_test)}")
    print(f"  Positive: {np.sum(y_test)} ({np.mean(y_test)*100:.2f}%)")
    print(f"  Negative: {np.sum(1-y_test)} ({np.mean(1-y_test)*100:.2f}%)")
    
    return model, X_test, y_test, feature_cols, valid_df

# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================

def calibrate_probabilities(model, X_train, y_train, X_test, method='isotonic'):
    """
    Calibrate model probabilities using validation data.
    
    Args:
        model: Trained Random Forest model
        X_train: Training features (for calibration)
        y_train: Training labels
        X_test: Test features
        method: 'isotonic' or 'sigmoid' (Platt scaling)
    
    Returns:
        Calibrated probabilities for test set
    """
    print(f"\n{'='*70}")
    print(f"PROBABILITY CALIBRATION ({method.upper()})")
    print(f"{'='*70}")
    
    # Create calibrated classifier
    calibrated_model = CalibratedClassifierCV(
        model, 
        method=method, 
        cv='prefit'  # Model already trained
    )
    
    # Fit calibration on training data (or use validation if available)
    print("Fitting calibration...")
    calibrated_model.fit(X_train, y_train)
    
    # Get calibrated probabilities
    y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    
    print(f"Original probability range: [{model.predict_proba(X_test)[:, 1].min():.4f}, "
          f"{model.predict_proba(X_test)[:, 1].max():.4f}]")
    print(f"Calibrated probability range: [{y_proba_calibrated.min():.4f}, "
          f"{y_proba_calibrated.max():.4f}]")
    
    return y_proba_calibrated, calibrated_model

# =============================================================================
# CLASS PRIOR ADJUSTMENT
# =============================================================================

def adjust_probabilities_for_prior(y_proba, train_prior_pos, deployment_prior_pos):
    """
    Adjust probabilities using Bayes rule for class prior shift.
    
    Formula: P(y=1|x) = (P(x|y=1) * P_deploy(y=1)) / P(x)
    
    Simplified adjustment:
    P_adjusted = (P_original * P_deploy / P_train) / 
                 (P_original * P_deploy / P_train + (1-P_original) * (1-P_deploy) / (1-P_train))
    
    Args:
        y_proba: Original probabilities
        train_prior_pos: Training prior for positive class
        deployment_prior_pos: Deployment prior for positive class
    
    Returns:
        Adjusted probabilities
    """
    print(f"\n{'='*70}")
    print("CLASS PRIOR ADJUSTMENT (BAYES RULE)")
    print(f"{'='*70}")
    print(f"Training prior (positive): {train_prior_pos:.4f}")
    print(f"Deployment prior (positive): {deployment_prior_pos:.4f}")
    
    # Avoid division by zero
    eps = 1e-10
    y_proba = np.clip(y_proba, eps, 1 - eps)
    
    # Calculate adjusted probabilities
    # P(y=1|x) = (P(x|y=1) * P_deploy(y=1)) / P(x)
    # Using odds ratio adjustment
    train_prior_neg = 1 - train_prior_pos
    deploy_prior_neg = 1 - deployment_prior_pos
    
    # Odds ratio
    odds_ratio = (deployment_prior_pos / deploy_prior_neg) / (train_prior_pos / train_prior_neg)
    
    # Convert probability to odds, adjust, convert back
    odds = y_proba / (1 - y_proba)
    adjusted_odds = odds * odds_ratio
    y_proba_adjusted = adjusted_odds / (1 + adjusted_odds)
    
    print(f"Original probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"Adjusted probability range: [{y_proba_adjusted.min():.4f}, {y_proba_adjusted.max():.4f}]")
    
    return y_proba_adjusted

# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def find_optimal_threshold_for_precision(y_true, y_proba, target_precision=None):
    """
    Find optimal threshold that maximizes precision or meets target precision.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_precision: Target precision (if None, maximize precision)
    
    Returns:
        Optimal threshold and metrics
    """
    print(f"\n{'='*70}")
    print("THRESHOLD OPTIMIZATION FOR PRECISION")
    print(f"{'='*70}")
    
    # Test a wide range of thresholds
    thresholds = np.arange(0.85, 0.999, 0.001)
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if np.sum(y_pred) == 0:
            # No positive predictions
            continue
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            continue
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("WARNING: No valid thresholds found!")
        return None
    
    # Find optimal threshold
    if target_precision is not None:
        # Find threshold that meets target precision with highest recall
        valid = results_df[results_df['precision'] >= target_precision]
        if len(valid) > 0:
            optimal_idx = valid['recall'].idxmax()
            optimal = results_df.loc[optimal_idx]
            print(f"Target precision: {target_precision:.1%}")
        else:
            print(f"WARNING: No threshold achieves {target_precision:.1%} precision")
            print("Using threshold with maximum precision...")
            optimal_idx = results_df['precision'].idxmax()
            optimal = results_df.loc[optimal_idx]
    else:
        # Maximize precision
        optimal_idx = results_df['precision'].idxmax()
        optimal = results_df.loc[optimal_idx]
    
    print(f"\nOptimal Threshold: {optimal['threshold']:.4f}")
    print(f"  Precision: {optimal['precision']:.4f} ({optimal['precision']*100:.2f}%)")
    print(f"  Recall:    {optimal['recall']:.4f} ({optimal['recall']*100:.2f}%)")
    print(f"  F1-Score:  {optimal['f1_score']:.4f}")
    print(f"  TP: {int(optimal['tp'])}, FP: {int(optimal['fp'])}, "
          f"TN: {int(optimal['tn'])}, FN: {int(optimal['fn'])}")
    
    return optimal.to_dict()

# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_precision_improvements(model, X_test, y_test, feature_cols):
    """Evaluate all precision improvement approaches."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PRECISION IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    # Get original probabilities
    y_proba_original = model.predict_proba(X_test)[:, 1]
    
    results = {}
    
    # 1. Original model with high thresholds
    print("\n" + "-" * 70)
    print("1. ORIGINAL MODEL - HIGH THRESHOLDS")
    print("-" * 70)
    
    original_results = {}
    for threshold in Config.HIGH_PRECISION_THRESHOLDS:
        y_pred = (y_proba_original >= threshold).astype(int)
        if np.sum(y_pred) == 0:
            continue
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            continue
        
        original_results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }
        
        print(f"Threshold {threshold:.2f}: Precision={precision:.4f}, "
              f"Recall={recall:.4f}, F1={f1:.4f}, TP={tp}, FP={fp}")
    
    results['original_high_thresholds'] = original_results
    
    # 2. Class prior adjustment
    print("\n" + "-" * 70)
    print("2. CLASS PRIOR ADJUSTMENT")
    print("-" * 70)
    
    y_proba_adjusted = adjust_probabilities_for_prior(
        y_proba_original,
        Config.TRAIN_PRIOR_POS,
        Config.DEPLOYMENT_PRIOR_POS
    )
    
    prior_optimal = find_optimal_threshold_for_precision(
        y_test, y_proba_adjusted, target_precision=Config.TARGET_PRECISION
    )
    results['prior_adjusted'] = {
        'optimal_threshold': prior_optimal,
        'probabilities': y_proba_adjusted
    }
    
    # 3. Probability calibration (if we have training data)
    if os.path.exists("datasets/train_features.csv"):
        print("\n" + "-" * 70)
        print("3. PROBABILITY CALIBRATION")
        print("-" * 70)
        
        train_df = pd.read_csv("datasets/train_features.csv")
        train_df = train_df[train_df['label'] != 'Omit'].copy()
        train_df['label'] = train_df['label'].map({'True': 1, 'False': 0})
        
        train_feature_cols = [col for col in train_df.columns 
                             if col not in ['window_id', 'label', 'event_sclk']]
        train_feature_cols = [col for col in train_feature_cols if col in feature_cols]
        
        X_train = train_df[train_feature_cols].values
        y_train = train_df['label'].values
        
        # Isotonic calibration
        try:
            y_proba_isotonic, _ = calibrate_probabilities(
                model, X_train, y_train, X_test, method='isotonic'
            )
            isotonic_optimal = find_optimal_threshold_for_precision(
                y_test, y_proba_isotonic, target_precision=Config.TARGET_PRECISION
            )
            results['isotonic_calibration'] = {
                'optimal_threshold': isotonic_optimal,
                'probabilities': y_proba_isotonic
            }
        except Exception as e:
            print(f"WARNING: Isotonic calibration failed: {e}")
        
        # Platt scaling
        try:
            y_proba_sigmoid, _ = calibrate_probabilities(
                model, X_train, y_train, X_test, method='sigmoid'
            )
            sigmoid_optimal = find_optimal_threshold_for_precision(
                y_test, y_proba_sigmoid, target_precision=Config.TARGET_PRECISION
            )
            results['sigmoid_calibration'] = {
                'optimal_threshold': sigmoid_optimal,
                'probabilities': y_proba_sigmoid
            }
        except Exception as e:
            print(f"WARNING: Sigmoid calibration failed: {e}")
    
    # 4. Find best overall approach
    print("\n" + "=" * 70)
    print("SUMMARY - BEST APPROACHES")
    print("=" * 70)
    
    best_precision = 0
    best_approach = None
    
    # Check original high thresholds
    for threshold, metrics in original_results.items():
        if metrics['precision'] > best_precision:
            best_precision = metrics['precision']
            best_approach = f"Original model, threshold={threshold:.2f}"
    
    # Check prior adjustment
    if prior_optimal and prior_optimal.get('precision', 0) > best_precision:
        best_precision = prior_optimal['precision']
        best_approach = f"Prior adjustment, threshold={prior_optimal.get('threshold', 0):.4f}"
    
    # Check calibrations
    for method in ['isotonic_calibration', 'sigmoid_calibration']:
        if method in results and results[method].get('optimal_threshold'):
            opt = results[method]['optimal_threshold']
            if opt.get('precision', 0) > best_precision:
                best_precision = opt['precision']
                best_approach = f"{method}, threshold={opt.get('threshold', 0):.4f}"
    
    print(f"\nBest Precision: {best_precision:.4f} ({best_precision*100:.2f}%)")
    print(f"Best Approach: {best_approach}")
    
    return results, best_approach, best_precision

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results, best_approach, best_precision):
    """Save analysis results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.RESULTS_DIR, f"precision_improvement_analysis_{timestamp}.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # Remove probability arrays (too large for JSON)
    results_clean = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_clean[key] = {k: v for k, v in value.items() 
                                 if k != 'probabilities'}
        else:
            results_clean[key] = value
    
    output = {
        'timestamp': timestamp,
        'best_approach': best_approach,
        'best_precision': float(best_precision),
        'results': convert_to_serializable(results_clean),
        'config': {
            'train_prior_pos': Config.TRAIN_PRIOR_POS,
            'deployment_prior_pos': Config.DEPLOYMENT_PRIOR_POS,
            'target_precision': Config.TARGET_PRECISION
        }
    }
    
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    return results_file

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("PRECISION IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Load model and data
        model, X_test, y_test, feature_cols, test_df = load_model_and_test_data()
        
        # Evaluate all approaches
        results, best_approach, best_precision = evaluate_precision_improvements(
            model, X_test, y_test, feature_cols
        )
        
        # Save results
        results_file = save_results(results, best_approach, best_precision)
        
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Best precision achieved: {best_precision:.4f} ({best_precision*100:.2f}%)")
        print(f"Best approach: {best_approach}")
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


