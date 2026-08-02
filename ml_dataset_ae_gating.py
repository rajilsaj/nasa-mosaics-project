#!/usr/bin/env python3
"""
ML Dataset Model with Autoencoder Gating
=========================================

This script implements AE gating for the ML dataset model:
1. Train a simple autoencoder on ML dataset pressure windows
2. Score all training windows with the autoencoder
3. Filter training data based on AE scores (keep top 50% or threshold-based)
4. Retrain RF model on filtered data
5. Evaluate on test set and compare to baseline

Expected improvement:
- Baseline: Precision 3.78%, Recall 6.58%, F1 4.80%
- Target: Precision 5-8%, Recall 7-10%, F1 6-8%
"""

import pandas as pd
import numpy as np
import os
import time
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for AE gating experiment."""
    
    # File paths
    TRAIN_FEATURES_FILE = "datasets/train_features.csv"
    VAL_FEATURES_FILE = "datasets/val_features.csv"
    TEST_FEATURES_FILE = "datasets/test_features.csv"
    TRAIN_WINDOWS_FILE = "datasets/train_windows.csv"  # For training autoencoder
    
    # Output directories
    MODELS_DIR = "models"
    RESULTS_DIR = "results"
    
    # Autoencoder configuration
    AE_HIDDEN_LAYERS = (32, 16, 32)  # Encoder-decoder architecture
    AE_MAX_ITER = 500
    AE_RANDOM_STATE = 42
    
    # Filtering strategy
    FILTER_METHOD = "top_percentile"  # Options: "top_percentile", "threshold"
    FILTER_PERCENTILE = 50  # Keep top 50% by AE score
    FILTER_THRESHOLD = None  # If using threshold method
    
    # Random Forest parameters (same as baseline)
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Feature columns (exclude metadata)
    EXCLUDE_COLUMNS = ['window_id', 'event_sclk', 'label']
    
    # Baseline metrics (from test results)
    BASELINE_METRICS = {
        'precision': 0.0378,
        'recall': 0.0658,
        'f1_score': 0.0480,
        'roc_auc': 0.7457
    }

# =============================================================================
# AUTOENCODER TRAINING
# =============================================================================

def load_training_windows():
    """Load training windows for autoencoder training."""
    print("=" * 70)
    print("LOADING TRAINING WINDOWS FOR AUTOENCODER")
    print("=" * 70)
    
    if not os.path.exists(Config.TRAIN_WINDOWS_FILE):
        print(f"[ERROR] Training windows file not found: {Config.TRAIN_WINDOWS_FILE}")
        print("[INFO] Trying to load from train_features.csv and reconstruct...")
        
        # Try to load from features file
        if os.path.exists(Config.TRAIN_FEATURES_FILE):
            print("[INFO] Cannot reconstruct windows from features. Need train_windows.csv")
            return None
        
        return None
    
    windows_df = pd.read_csv(Config.TRAIN_WINDOWS_FILE)
    print(f"Loaded {len(windows_df):,} window samples")
    
    # Check for PRESSURE column
    if 'PRESSURE' not in windows_df.columns:
        print("[ERROR] PRESSURE column not found in training windows")
        return None
    
    # Group by window_id to get individual windows
    if 'window_id' in windows_df.columns:
        window_groups = windows_df.groupby('window_id')
        print(f"Found {len(window_groups):,} unique windows")
    else:
        print("[WARNING] No window_id found, assuming sequential windows of size 60")
        window_groups = []
        window_size = 60
        for i in range(0, len(windows_df), window_size):
            window_data = windows_df.iloc[i:i+window_size]
            if len(window_data) == window_size:
                window_groups.append((i // window_size, window_data))
    
    return window_groups

def train_autoencoder(window_groups):
    """
    Train a simple autoencoder on pressure windows.
    
    Args:
        window_groups: Grouped windows by window_id
        
    Returns:
        Trained autoencoder model and scaler
    """
    print("\n" + "=" * 70)
    print("TRAINING AUTOENCODER")
    print("=" * 70)
    
    # Extract pressure windows
    pressure_windows = []
    window_ids = []
    
    print("Extracting pressure windows...")
    for window_id, window_data in window_groups:
        if 'PRESSURE' in window_data.columns:
            pressure = window_data['PRESSURE'].values
            if len(pressure) == 60:  # Standard window size
                pressure_windows.append(pressure)
                window_ids.append(window_id)
    
    if len(pressure_windows) == 0:
        print("[ERROR] No valid pressure windows found!")
        return None, None
    
    X_ae = np.array(pressure_windows)
    print(f"Extracted {len(X_ae):,} pressure windows of size {X_ae.shape[1]}")
    
    # Normalize pressure values
    scaler = StandardScaler()
    X_ae_scaled = scaler.fit_transform(X_ae)
    
    # Train autoencoder (simple MLP)
    print(f"\nTraining autoencoder...")
    print(f"  Architecture: {X_ae.shape[1]} -> {Config.AE_HIDDEN_LAYERS} -> {X_ae.shape[1]}")
    print(f"  Max iterations: {Config.AE_MAX_ITER}")
    
    start_time = time.time()
    
    autoencoder = MLPRegressor(
        hidden_layer_sizes=Config.AE_HIDDEN_LAYERS,
        max_iter=Config.AE_MAX_ITER,
        random_state=Config.AE_RANDOM_STATE,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    autoencoder.fit(X_ae_scaled, X_ae_scaled)
    
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f} seconds")
    
    # Test reconstruction
    X_reconstructed = autoencoder.predict(X_ae_scaled)
    reconstruction_error = np.mean((X_ae_scaled - X_reconstructed) ** 2)
    print(f"  Mean reconstruction error: {reconstruction_error:.6f}")
    
    return autoencoder, scaler

def score_windows_with_ae(autoencoder, scaler, window_groups):
    """
    Score all windows with the autoencoder (reconstruction error).
    
    Lower reconstruction error = more "normal" (less anomalous)
    Higher reconstruction error = more "anomalous" (potential vortex)
    
    Args:
        autoencoder: Trained autoencoder model
        scaler: Fitted scaler
        window_groups: Grouped windows
        
    Returns:
        Dictionary mapping window_id to AE score
    """
    print("\n" + "=" * 70)
    print("SCORING WINDOWS WITH AUTOENCODER")
    print("=" * 70)
    
    window_scores = {}
    
    print("Computing reconstruction errors...")
    for window_id, window_data in window_groups:
        if 'PRESSURE' in window_data.columns:
            pressure = window_data['PRESSURE'].values
            if len(pressure) == 60:
                # Normalize and reconstruct
                pressure_scaled = scaler.transform(pressure.reshape(1, -1))
                reconstructed = autoencoder.predict(pressure_scaled)
                
                # Compute reconstruction error (MSE)
                reconstruction_error = np.mean((pressure_scaled - reconstructed) ** 2)
                
                # Higher error = more anomalous = higher score
                window_scores[window_id] = float(reconstruction_error)
    
    print(f"Scored {len(window_scores):,} windows")
    print(f"  Score range: {min(window_scores.values()):.6f} - {max(window_scores.values()):.6f}")
    print(f"  Mean score: {np.mean(list(window_scores.values())):.6f}")
    print(f"  Std score: {np.std(list(window_scores.values())):.6f}")
    
    return window_scores

# =============================================================================
# DATA FILTERING
# =============================================================================

def filter_training_data(train_df, window_scores):
    """
    Filter training data based on AE scores.
    
    Args:
        train_df: Training features DataFrame
        window_scores: Dictionary of window_id -> AE score
        
    Returns:
        Filtered training DataFrame
    """
    print("\n" + "=" * 70)
    print("FILTERING TRAINING DATA")
    print("=" * 70)
    
    original_size = len(train_df)
    print(f"Original training size: {original_size:,} samples")
    
    # Add AE scores to dataframe
    if 'window_id' not in train_df.columns:
        print("[ERROR] window_id column not found in training data!")
        return None
    
    train_df = train_df.copy()
    train_df['ae_score'] = train_df['window_id'].map(window_scores)
    
    # Check how many have scores
    scored_count = train_df['ae_score'].notna().sum()
    print(f"Samples with AE scores: {scored_count:,} ({scored_count/original_size*100:.1f}%)")
    
    if scored_count == 0:
        print("[ERROR] No samples have AE scores! Cannot filter.")
        return None
    
    # Filter based on method
    if Config.FILTER_METHOD == "top_percentile":
        threshold = np.percentile(train_df['ae_score'].dropna(), 
                                  100 - Config.FILTER_PERCENTILE)
        filtered_df = train_df[train_df['ae_score'] >= threshold].copy()
        print(f"Filtering method: Top {Config.FILTER_PERCENTILE}% by AE score")
        print(f"  Threshold: {threshold:.6f}")
        
    elif Config.FILTER_METHOD == "threshold":
        if Config.FILTER_THRESHOLD is None:
            # Use median as default
            Config.FILTER_THRESHOLD = train_df['ae_score'].median()
        filtered_df = train_df[train_df['ae_score'] >= Config.FILTER_THRESHOLD].copy()
        print(f"Filtering method: Threshold-based")
        print(f"  Threshold: {Config.FILTER_THRESHOLD:.6f}")
    else:
        print(f"[ERROR] Unknown filter method: {Config.FILTER_METHOD}")
        return None
    
    filtered_size = len(filtered_df)
    print(f"Filtered training size: {filtered_size:,} samples")
    print(f"  Reduction: {original_size - filtered_size:,} samples ({100*(1-filtered_size/original_size):.1f}%)")
    
    # Check class distribution
    if 'label' in filtered_df.columns:
        class_dist = filtered_df['label'].value_counts()
        print(f"\nFiltered class distribution:")
        print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/filtered_size*100:.1f}%)")
        print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/filtered_size*100:.1f}%)")
    
    # Remove AE score column (not a feature)
    filtered_df = filtered_df.drop(columns=['ae_score'])
    
    return filtered_df

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_rf_model(X_train, y_train):
    """Train Random Forest model on filtered data."""
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    print(f"\nRF Parameters: {Config.RF_PARAMS}")
    
    start_time = time.time()
    
    rf_model = RandomForestClassifier(**Config.RF_PARAMS)
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    
    return rf_model, training_time

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X, y, split_name="Test", feature_cols=None):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print(f"EVALUATION - {split_name.upper()} SET")
    print("=" * 70)
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nPerformance Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {tn:8d}  {fp:8d}")
    print(f"Positive  {fn:8d}  {tp:8d}")
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }

def compare_to_baseline(new_metrics, baseline_metrics):
    """Compare new metrics to baseline."""
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    
    print(f"\n{'Metric':<15} {'Baseline':<12} {'New Model':<12} {'Change':<12} {'Status':<10}")
    print("-" * 70)
    
    metrics = ['precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics:
        baseline_val = baseline_metrics.get(metric, 0)
        new_val = new_metrics.get(metric, 0)
        change = new_val - baseline_val
        change_pct = (change / baseline_val * 100) if baseline_val > 0 else 0
        
        if change > 0:
            status = "[IMPROVED]"
        elif change < 0:
            status = "[WORSE]"
        else:
            status = "[SAME]"
        
        print(f"{metric.capitalize():<15} {baseline_val:<12.4f} {new_val:<12.4f} "
              f"{change:+.4f} ({change_pct:+.1f}%)  {status}")
    
    # Overall verdict
    print("\n" + "=" * 70)
    if new_metrics['f1_score'] > baseline_metrics['f1_score']:
        print("[SUCCESS] OVERALL: IMPROVEMENT DETECTED")
    else:
        print("[FAILED] OVERALL: NO IMPROVEMENT (or worse)")
    print("=" * 70)

# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(autoencoder, scaler, rf_model, window_scores, 
                train_metrics, test_metrics, training_time):
    """Save all models and results."""
    print("\n" + "=" * 70)
    print("SAVING MODELS AND RESULTS")
    print("=" * 70)
    
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save autoencoder
    ae_path = os.path.join(Config.MODELS_DIR, f"ae_ml_dataset_{timestamp}.pkl")
    joblib.dump({'model': autoencoder, 'scaler': scaler}, ae_path)
    print(f"Autoencoder saved: {ae_path}")
    
    # Save RF model
    rf_path = os.path.join(Config.MODELS_DIR, f"rf_ae_gated_ml_{timestamp}.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"RF model saved: {rf_path}")
    
    # Save window scores
    scores_path = os.path.join(Config.RESULTS_DIR, f"ae_window_scores_{timestamp}.json")
    with open(scores_path, 'w') as f:
        json.dump(window_scores, f, indent=2)
    print(f"Window scores saved: {scores_path}")
    
    # Save results summary
    results = {
        'timestamp': timestamp,
        'experiment': 'ML Dataset + AE Gating',
        'filter_method': Config.FILTER_METHOD,
        'filter_percentile': Config.FILTER_PERCENTILE,
        'baseline_metrics': Config.BASELINE_METRICS,
        'new_metrics': {
            'training': train_metrics,
            'test': test_metrics
        },
        'training_time_seconds': training_time,
        'improvement': {
            'precision_delta': test_metrics['precision'] - Config.BASELINE_METRICS['precision'],
            'recall_delta': test_metrics['recall'] - Config.BASELINE_METRICS['recall'],
            'f1_delta': test_metrics['f1_score'] - Config.BASELINE_METRICS['f1_score'],
            'roc_auc_delta': test_metrics['roc_auc'] - Config.BASELINE_METRICS['roc_auc']
        }
    }
    
    results_path = os.path.join(Config.RESULTS_DIR, f"ae_gating_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")
    
    return results_path

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main execution pipeline."""
    print("=" * 70)
    print("ML DATASET MODEL WITH AUTOENCODER GATING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baseline: Precision {Config.BASELINE_METRICS['precision']:.4f}, "
          f"F1 {Config.BASELINE_METRICS['f1_score']:.4f}")
    print("=" * 70)
    
    # Step 1: Load training windows for autoencoder
    window_groups = load_training_windows()
    if window_groups is None:
        print("\n[ERROR] Failed to load training windows. Exiting.")
        return 1
    
    # Step 2: Train autoencoder
    autoencoder, scaler = train_autoencoder(window_groups)
    if autoencoder is None:
        print("\n[ERROR] Failed to train autoencoder. Exiting.")
        return 1
    
    # Step 3: Score windows with autoencoder
    window_scores = score_windows_with_ae(autoencoder, scaler, window_groups)
    if not window_scores:
        print("\n[ERROR] Failed to score windows. Exiting.")
        return 1
    
    # Step 4: Load training features
    print("\n" + "=" * 70)
    print("LOADING TRAINING FEATURES")
    print("=" * 70)
    
    if not os.path.exists(Config.TRAIN_FEATURES_FILE):
        print(f"[ERROR] Training features file not found: {Config.TRAIN_FEATURES_FILE}")
        return 1
    
    train_df = pd.read_csv(Config.TRAIN_FEATURES_FILE)
    print(f"Loaded {len(train_df):,} training feature vectors")
    
    # Step 5: Filter training data
    filtered_train_df = filter_training_data(train_df, window_scores)
    if filtered_train_df is None:
        print("\n[ERROR] Failed to filter training data. Exiting.")
        return 1
    
    # Step 6: Prepare features and labels
    feature_cols = [col for col in filtered_train_df.columns 
                   if col not in Config.EXCLUDE_COLUMNS]
    X_train = filtered_train_df[feature_cols].values
    y_train = filtered_train_df['label'].values
    
    print(f"\nPrepared training data:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {feature_cols}")
    
    # Step 7: Train RF model
    rf_model, training_time = train_rf_model(X_train, y_train)
    
    # Step 8: Evaluate on training set
    train_metrics = evaluate_model(rf_model, X_train, y_train, "Training", feature_cols)
    
    # Step 9: Load and evaluate on test set
    print("\n" + "=" * 70)
    print("LOADING TEST SET")
    print("=" * 70)
    
    if not os.path.exists(Config.TEST_FEATURES_FILE):
        print(f"[ERROR] Test features file not found: {Config.TEST_FEATURES_FILE}")
        return 1
    
    test_df = pd.read_csv(Config.TEST_FEATURES_FILE)
    print(f"Loaded {len(test_df):,} test feature vectors")
    
    # Prepare test features (same features as training)
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values
    
    # Step 10: Evaluate on test set
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test", feature_cols)
    
    # Step 11: Compare to baseline
    compare_to_baseline(test_metrics, Config.BASELINE_METRICS)
    
    # Step 12: Save results
    results_path = save_results(
        autoencoder, scaler, rf_model, window_scores,
        train_metrics, test_metrics, training_time
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    print(f"\nResults saved to: {results_path}")
    print(f"\nSummary:")
    print(f"  Baseline F1: {Config.BASELINE_METRICS['f1_score']:.4f}")
    print(f"  New F1:      {test_metrics['f1_score']:.4f}")
    print(f"  Improvement: {test_metrics['f1_score'] - Config.BASELINE_METRICS['f1_score']:+.4f}")
    
    return 0

if __name__ == "__main__":
    exit(main())
