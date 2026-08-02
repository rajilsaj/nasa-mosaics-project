#!/usr/bin/env python3
"""
Comprehensive Dataset Model with Autoencoder Gating
====================================================

This script implements AE gating for the comprehensive dataset baseline model:
1. Train a simple autoencoder on comprehensive dataset pressure windows
2. Score all training windows with the autoencoder
3. Filter training data based on AE scores (keep top 50% or threshold-based)
4. Retrain RF model on filtered data (15 features, baseline model)
5. Evaluate on test set and compare to baseline

Expected improvement:
- Baseline: Precision 1.28%, Recall 8.37%, F1 2.22%, ROC AUC 0.5050
- Target: Precision 2-4%, Recall 10-15%, F1 3-5%
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
    
    # File paths (comprehensive dataset)
    COMPREHENSIVE_DIR = "comprehensive_analysis"
    DATA_DIR = os.path.join(COMPREHENSIVE_DIR, "data")
    TRAIN_WINDOWS_FILE = os.path.join(DATA_DIR, "windows", "datasets/train_windows.csv")
    TRAIN_FEATURES_FILE = os.path.join(DATA_DIR, "features", "datasets/train_balanced.csv")
    TRAIN_SPLIT_FILE = os.path.join(DATA_DIR, "splits", "ml_train.csv")  # For extracting negative windows
    TEST_SLIDING_FEATURES = os.path.join(DATA_DIR, "features", "test_sliding_features_step10.csv")
    
    # Use existing autoencoder features for gating (instead of training new AE)
    USE_EXISTING_AE_FEATURES = True  # Set to False to train new autoencoder
    
    # Output directories
    MODELS_DIR = os.path.join(COMPREHENSIVE_DIR, "models")
    RESULTS_DIR = os.path.join(COMPREHENSIVE_DIR, "results")
    
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
        'n_estimators': 200,        # More trees for comprehensive dataset
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1,
        'oob_score': True
    }
    
    # Feature columns (exclude metadata) - baseline model uses 15 features
    EXCLUDE_COLUMNS = ['window_id', 'event_sclk', 'label', 'sliding_window_id',
                       'sliding_start_idx', 'sliding_end_idx', 'sliding_start_sclk',
                       'sliding_end_sclk']
    
    # Baseline metrics (from comprehensive dataset baseline model)
    BASELINE_METRICS = {
        'precision': 0.0128,
        'recall': 0.0837,
        'f1_score': 0.0222,
        'roc_auc': 0.5050
    }

# =============================================================================
# AUTOENCODER TRAINING
# =============================================================================

def extract_negative_windows(ml_train_df, positive_windows_df, num_negative, window_size=60):
    """Extract negative windows from safe regions."""
    print("Extracting negative windows from safe regions...")
    
    # Create forbidden zones (around positive windows)
    forbidden = np.zeros(len(ml_train_df), dtype=bool)
    buffer = 50  # Buffer around positive events
    
    for window_id, window_data in positive_windows_df.groupby('window_id'):
        # Find indices in ml_train_df that correspond to this window
        window_sclks = window_data['SCLK'].values
        for sclk in window_sclks:
            matches = ml_train_df[ml_train_df['SCLK'] == sclk].index
            if len(matches) > 0:
                idx = matches[0]
                start = max(0, idx - buffer)
                end = min(len(ml_train_df), idx + window_size + buffer)
                forbidden[start:end] = True
    
    # Find safe starting positions
    valid_starts = []
    for i in range(len(ml_train_df) - window_size + 1):
        if not forbidden[i:i + window_size].any():
            valid_starts.append(i)
    
    print(f"  Found {len(valid_starts):,} safe starting positions")
    
    if len(valid_starts) < num_negative:
        print(f"  [WARNING] Only {len(valid_starts)} safe positions, reducing to {len(valid_starts)}")
        num_negative = len(valid_starts)
    
    # Sample negative windows
    np.random.seed(42)
    sampled_starts = np.random.choice(valid_starts, size=num_negative, replace=False)
    
    negative_windows = []
    for neg_id, start_idx in enumerate(sampled_starts):
        window = ml_train_df.iloc[start_idx:start_idx + window_size].copy()
        window['window_id'] = neg_id + 100000  # Offset to avoid conflicts
        negative_windows.append(window)
    
    return negative_windows

def load_training_windows():
    """Load training windows for autoencoder training (both positive and negative)."""
    print("=" * 70)
    print("LOADING TRAINING WINDOWS FOR AUTOENCODER")
    print("=" * 70)
    
    # Load positive windows
    if not os.path.exists(Config.TRAIN_WINDOWS_FILE):
        print(f"[ERROR] Training windows file not found: {Config.TRAIN_WINDOWS_FILE}")
        return None
    
    positive_windows_df = pd.read_csv(Config.TRAIN_WINDOWS_FILE)
    print(f"Loaded {len(positive_windows_df):,} positive window samples")
    
    # Load raw ML data to extract negative windows
    if not os.path.exists(Config.TRAIN_SPLIT_FILE):
        print(f"[ERROR] Training split file not found: {Config.TRAIN_SPLIT_FILE}")
        print("[INFO] Will use only positive windows (limited effectiveness)")
    else:
        ml_train_df = pd.read_csv(Config.TRAIN_SPLIT_FILE)
        print(f"Loaded {len(ml_train_df):,} raw ML training samples")
        
        # Extract negative windows (same number as positive)
        num_positive = positive_windows_df['window_id'].nunique()
        negative_windows = extract_negative_windows(ml_train_df, positive_windows_df, num_positive)
        
        if negative_windows:
            negative_windows_df = pd.concat(negative_windows, ignore_index=True)
            print(f"Extracted {len(negative_windows):,} negative windows")
            
            # Combine positive and negative
            all_windows_df = pd.concat([positive_windows_df, negative_windows_df], ignore_index=True)
            print(f"Total windows: {len(all_windows_df.groupby('window_id')):,} (positive + negative)")
            
            # Group by window_id
            window_groups = all_windows_df.groupby('window_id')
            return window_groups
    
    # Fallback: use only positive windows
    if 'PRESSURE' not in positive_windows_df.columns:
        print("[ERROR] PRESSURE column not found in training windows")
        return None
    
    window_groups = positive_windows_df.groupby('window_id')
    print(f"[WARNING] Using only positive windows ({len(window_groups):,} windows)")
    print(f"[WARNING] AE may not distinguish anomalies well without negative examples")
    
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

def score_windows_with_existing_ae_features(train_df):
    """
    Score windows using existing autoencoder features in comprehensive dataset.
    
    Uses autoencoder_window_hits_mean as the score (higher = more anomalous).
    Returns scores for ALL samples in train_df (not just by window_id).
    
    Args:
        train_df: Training features DataFrame with autoencoder features
        
    Returns:
        Series with AE scores indexed by DataFrame index (for direct filtering)
    """
    print("\n" + "=" * 70)
    print("SCORING WINDOWS WITH EXISTING AUTOENCODER FEATURES")
    print("=" * 70)
    
    if 'autoencoder_window_hits_mean' not in train_df.columns:
        print("[ERROR] autoencoder_window_hits_mean not found in training data!")
        print("[INFO] Available columns:", list(train_df.columns))
        return None
    
    # Use autoencoder_window_hits_mean as score (higher = more anomalous)
    # Handle NaN values (negative samples might not have AE features)
    ae_scores = train_df['autoencoder_window_hits_mean'].fillna(0.0)
    
    print(f"Scored {len(ae_scores):,} samples using existing AE features")
    valid_scores = ae_scores[ae_scores.notna()]
    if len(valid_scores) > 0:
        print(f"  Score range: {valid_scores.min():.2f} - {valid_scores.max():.2f}")
        print(f"  Mean score: {valid_scores.mean():.2f}")
        print(f"  Std score: {valid_scores.std():.2f}")
        print(f"  Samples with NaN: {(ae_scores.isna()).sum()}")
    
    return ae_scores

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
    if window_scores:
        print(f"  Score range: {min(window_scores.values()):.6f} - {max(window_scores.values()):.6f}")
        print(f"  Mean score: {np.mean(list(window_scores.values())):.6f}")
        print(f"  Std score: {np.std(list(window_scores.values())):.6f}")
    
    return window_scores

# =============================================================================
# DATA FILTERING
# =============================================================================

def filter_training_data(train_df, ae_scores):
    """
    Filter training data based on AE scores.
    
    Args:
        train_df: Training features DataFrame
        ae_scores: Series with AE scores (indexed by DataFrame index)
        
    Returns:
        Filtered training DataFrame
    """
    print("\n" + "=" * 70)
    print("FILTERING TRAINING DATA")
    print("=" * 70)
    
    original_size = len(train_df)
    print(f"Original training size: {original_size:,} samples")
    
    # Add AE scores to dataframe (using index alignment)
    train_df = train_df.copy()
    train_df['ae_score'] = ae_scores.values
    
    # Check how many have scores
    scored_count = train_df['ae_score'].notna().sum()
    print(f"Samples with AE scores: {scored_count:,} ({scored_count/original_size*100:.1f}%)")
    
    if scored_count == 0:
        print("[ERROR] No samples have AE scores! Cannot filter.")
        return None
    
    # Filter based on method - ensure we keep both classes
    if Config.FILTER_METHOD == "top_percentile":
        # Filter separately for each class to maintain balance
        positive_df = train_df[train_df['label'] == 1].copy()
        negative_df = train_df[train_df['label'] == 0].copy()
        
        if len(positive_df) > 0 and len(negative_df) > 0:
            # Keep top percentile from each class
            pos_scores = positive_df['ae_score'].dropna()
            neg_scores = negative_df['ae_score'].dropna()
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                pos_threshold = np.percentile(pos_scores, 100 - Config.FILTER_PERCENTILE)
                neg_threshold = np.percentile(neg_scores, 100 - Config.FILTER_PERCENTILE)
                
                filtered_pos = positive_df[positive_df['ae_score'] >= pos_threshold].copy()
                filtered_neg = negative_df[negative_df['ae_score'] >= neg_threshold].copy()
                
                filtered_df = pd.concat([filtered_pos, filtered_neg], ignore_index=True)
                print(f"Filtering method: Top {Config.FILTER_PERCENTILE}% by AE score (per class)")
                print(f"  Positive threshold: {pos_threshold:.2f}")
                print(f"  Negative threshold: {neg_threshold:.2f}")
            else:
                # Fallback if one class has no scores
                threshold = np.percentile(train_df['ae_score'].dropna(), 
                                          100 - Config.FILTER_PERCENTILE)
                filtered_df = train_df[train_df['ae_score'] >= threshold].copy()
                print(f"Filtering method: Top {Config.FILTER_PERCENTILE}% by AE score")
                print(f"  Threshold: {threshold:.2f}")
        else:
            # Fallback to original method if only one class
            threshold = np.percentile(train_df['ae_score'].dropna(), 
                                      100 - Config.FILTER_PERCENTILE)
            filtered_df = train_df[train_df['ae_score'] >= threshold].copy()
            print(f"Filtering method: Top {Config.FILTER_PERCENTILE}% by AE score")
            print(f"  Threshold: {threshold:.2f}")
        
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
    
    # Handle single-class case
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except IndexError:
        # Model only has one class, predict_proba returns single column
        y_proba = model.predict_proba(X)[:, 0]
        print("[WARNING] Model has only one class. Using single-column probabilities.")
    
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

def save_results(autoencoder, scaler, rf_model, ae_scores, 
                train_metrics, test_metrics, training_time):
    """Save all models and results."""
    print("\n" + "=" * 70)
    print("SAVING MODELS AND RESULTS")
    print("=" * 70)
    
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save autoencoder
    ae_path = os.path.join(Config.MODELS_DIR, f"ae_comprehensive_{timestamp}.pkl")
    joblib.dump({'model': autoencoder, 'scaler': scaler}, ae_path)
    print(f"Autoencoder saved: {ae_path}")
    
    # Save RF model
    rf_path = os.path.join(Config.MODELS_DIR, f"rf_ae_gated_comprehensive_{timestamp}.pkl")
    joblib.dump(rf_model, rf_path)
    print(f"RF model saved: {rf_path}")
    
    # Save window scores (convert Series to dict if needed)
    if isinstance(ae_scores, pd.Series):
        scores_dict = ae_scores.to_dict()
    else:
        scores_dict = ae_scores
    
    scores_path = os.path.join(Config.RESULTS_DIR, f"ae_window_scores_comprehensive_{timestamp}.json")
    with open(scores_path, 'w') as f:
        json.dump(scores_dict, f, indent=2)
    print(f"Window scores saved: {scores_path}")
    
    # Save results summary
    results = {
        'timestamp': timestamp,
        'experiment': 'Comprehensive Dataset + AE Gating',
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
    
    results_path = os.path.join(Config.RESULTS_DIR, f"ae_gating_comprehensive_results_{timestamp}.json")
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
    print("COMPREHENSIVE DATASET MODEL WITH AUTOENCODER GATING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Baseline: Precision {Config.BASELINE_METRICS['precision']:.4f}, "
          f"F1 {Config.BASELINE_METRICS['f1_score']:.4f}")
    print("=" * 70)
    
    # Step 1: Load training features
    print("\n" + "=" * 70)
    print("LOADING TRAINING FEATURES")
    print("=" * 70)
    
    if not os.path.exists(Config.TRAIN_FEATURES_FILE):
        print(f"[ERROR] Training features file not found: {Config.TRAIN_FEATURES_FILE}")
        return 1
    
    train_df = pd.read_csv(Config.TRAIN_FEATURES_FILE)
    print(f"Loaded {len(train_df):,} training feature vectors")
    
    # Step 2: Choose scoring method
    autoencoder = None
    scaler = None
    
    if Config.USE_EXISTING_AE_FEATURES and 'autoencoder_window_hits_mean' in train_df.columns:
        print("\n[INFO] Using existing autoencoder features for gating")
        print("[INFO] This is more efficient and uses the pre-computed AE features")
        
        # Score using existing AE features
        ae_scores = score_windows_with_existing_ae_features(train_df)
        
        if ae_scores is None:
            print("\n[ERROR] Failed to score windows with existing AE features. Exiting.")
            return 1
    else:
        print("\n[INFO] Training new autoencoder for gating")
        
        # Load training windows for autoencoder
        window_groups = load_training_windows()
        if window_groups is None:
            print("\n[ERROR] Failed to load training windows. Exiting.")
            return 1
        
        # Train autoencoder
        autoencoder, scaler = train_autoencoder(window_groups)
        if autoencoder is None:
            print("\n[ERROR] Failed to train autoencoder. Exiting.")
            return 1
        
        # Score windows with autoencoder
        window_scores_dict = score_windows_with_ae(autoencoder, scaler, window_groups)
        if not window_scores_dict:
            print("\n[ERROR] Failed to score windows. Exiting.")
            return 1
        
        # Convert dict to Series for filtering
        train_df['ae_score_temp'] = train_df['window_id'].map(window_scores_dict)
        ae_scores = train_df['ae_score_temp']
        train_df = train_df.drop(columns=['ae_score_temp'])
    
    # Step 5: Filter training data
    filtered_train_df = filter_training_data(train_df, ae_scores)
    if filtered_train_df is None:
        print("\n[ERROR] Failed to filter training data. Exiting.")
        return 1
    
    # Step 6: Prepare features and labels (use only 15 baseline features)
    # Exclude autoencoder features to match baseline model
    all_feature_cols = [col for col in filtered_train_df.columns 
                       if col not in Config.EXCLUDE_COLUMNS]
    
    # Filter to only baseline 15 features (exclude autoencoder features)
    baseline_features = [
        'overall_slope', 'first_half_slope', 'second_half_slope', 'trend_consistency',
        'pressure_drop', 'drop_rate', 'min_position',
        'mean', 'std', 'range',
        'first_half_mean', 'second_half_mean', 'mean_ratio',
        'min_zscore', 'anomaly_strength'
    ]
    
    feature_cols = [f for f in baseline_features if f in all_feature_cols]
    
    if len(feature_cols) < 10:
        print(f"[WARNING] Only {len(feature_cols)} baseline features found. Using all available features.")
        feature_cols = all_feature_cols
    
    X_train = filtered_train_df[feature_cols].values
    y_train = filtered_train_df['label'].values
    
    print(f"\nPrepared training data:")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X_train):,}")
    print(f"  Features: {feature_cols}")
    
    # Step 7: Train RF model
    rf_model, training_time = train_rf_model(X_train, y_train)
    
    # Check if we have both classes
    unique_classes = np.unique(y_train)
    if len(unique_classes) == 1:
        print("\n[WARNING] Only one class in training data after filtering!")
        print(f"Class: {unique_classes[0]} (1=positive, 0=negative)")
        print("Model will predict only one class. This may not be useful.")
    
    # Step 9: Load and evaluate on test set (sliding windows)
    print("\n" + "=" * 70)
    print("LOADING TEST SET (SLIDING WINDOWS)")
    print("=" * 70)
    
    if not os.path.exists(Config.TEST_SLIDING_FEATURES):
        print(f"[ERROR] Test sliding features file not found: {Config.TEST_SLIDING_FEATURES}")
        return 1
    
    test_df = pd.read_csv(Config.TEST_SLIDING_FEATURES)
    print(f"Loaded {len(test_df):,} test sliding window feature vectors")
    
    # Filter out 'Omit' labels and convert to binary
    valid_mask = test_df['label'] != 'Omit'
    valid_df = test_df[valid_mask].copy()
    
    # Convert labels properly
    label_map = {'True': 1, 'False': 0, True: 1, False: 0, 1: 1, 0: 0}
    valid_df['label'] = valid_df['label'].map(label_map)
    valid_df = valid_df[valid_df['label'].notna()].copy()  # Remove any unmapped labels
    
    # Prepare test features (same features as training)
    X_test = valid_df[feature_cols].values
    y_test = valid_df['label'].values.astype(int)
    
    print(f"Valid test samples: {len(y_test):,}")
    unique, counts = np.unique(y_test, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Class distribution:")
    for label, count in sorted(class_dist.items()):
        label_name = 'Positive' if label == 1 else 'Negative'
        print(f"  {label_name} ({label}): {count:,} ({count/len(y_test)*100:.2f}%)")
    
    # Check if model has only one class (all positive after filtering)
    if len(np.unique(y_train)) == 1:
        print("\n[WARNING] Model trained on single class only. Cannot evaluate properly.")
        print("This happens when AE filtering removes all negative samples.")
        print("Skipping training evaluation, proceeding to test set...")
        train_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0,
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
        }
    else:
        # Step 8: Evaluate on training set
        train_metrics = evaluate_model(rf_model, X_train, y_train, "Training", feature_cols)
    
    # Step 10: Evaluate on test set
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test (Sliding Windows)", feature_cols)
    
    # Step 11: Compare to baseline
    compare_to_baseline(test_metrics, Config.BASELINE_METRICS)
    
    # Step 12: Save results
    results_path = save_results(
        autoencoder, scaler, rf_model, ae_scores,
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
