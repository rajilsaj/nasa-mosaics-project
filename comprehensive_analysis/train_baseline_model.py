#!/usr/bin/env python3
"""
Baseline Random Forest Training - Comprehensive Dataset
=======================================================

As a seasoned RF scientist, this script:
1. Trains baseline RF model on balanced training data (1:1)
2. Validates on natural imbalance (realistic deployment scenario)
3. Tracks comprehensive metrics (F1, Precision, Recall, ROC AUC)
4. Saves model and metadata for future comparison

Phase 3.1: Baseline Model (15 original features)
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score, 
    roc_curve, precision_recall_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Random Forest hyperparameters (from expert guide)
RF_CONFIG = {
    'n_estimators': 200,        # More trees = better
    'max_depth': 15,            # Prevent overfitting
    'min_samples_split': 10,    # Require sufficient samples
    'min_samples_leaf': 5,      # Minimum leaf size
    'max_features': 'sqrt',     # Standard
    'class_weight': 'balanced', # Handle imbalance
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': True           # Out-of-bag validation
}

# Feature selection: Use only 15 original features (exclude autoencoder for baseline)
ORIGINAL_FEATURES = [
    'overall_slope', 'first_half_slope', 'second_half_slope', 'trend_consistency',
    'pressure_drop', 'drop_rate', 'min_position',
    'mean', 'std', 'range',
    'first_half_mean', 'second_half_mean', 'mean_ratio',
    'min_zscore', 'anomaly_strength'
]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data():
    """Load balanced training data."""
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)
    
    train_file = os.path.join(FEATURES_DIR, "train_balanced.csv")
    
    if not os.path.exists(train_file):
        print(f"[ERROR] Training file not found: {train_file}")
        print("[INFO] Run negative_sampling.py --split train --ratio 1.0 first!")
        return None, None, None
    
    train_df = pd.read_csv(train_file)
    print(f"Loaded {len(train_df):,} training samples")
    
    # Check class distribution
    if 'label' in train_df.columns:
        class_dist = train_df['label'].value_counts()
        print(f"Class distribution:")
        print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(train_df)*100:.1f}%)")
        print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(train_df)*100:.1f}%)")
    
    # Select features (only original 15, exclude autoencoder)
    available_features = [f for f in ORIGINAL_FEATURES if f in train_df.columns]
    missing_features = [f for f in ORIGINAL_FEATURES if f not in train_df.columns]
    
    if missing_features:
        print(f"[WARNING] Missing features: {missing_features}")
    
    if len(available_features) < 10:
        print(f"[ERROR] Too few features available: {len(available_features)}")
        return None, None, None
    
    print(f"Using {len(available_features)} original features (baseline)")
    
    # Prepare features and labels
    X_train = train_df[available_features].values
    y_train = train_df['label'].values
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return X_train, y_train, available_features

def load_validation_data(feature_cols):
    """Load validation data (natural imbalance or balanced for evaluation)."""
    print("\n" + "=" * 70)
    print("LOADING VALIDATION DATA")
    print("=" * 70)
    
    # Try balanced first (if available), then fall back to features only
    val_balanced_file = os.path.join(FEATURES_DIR, "val_balanced.csv")
    val_file = os.path.join(FEATURES_DIR, "val_features.csv")
    
    if os.path.exists(val_balanced_file):
        val_file = val_balanced_file
        print(f"Using balanced validation set (for proper evaluation)")
    else:
        print(f"Using features-only validation set (natural imbalance)")
    
    val_file = val_file
    
    if not os.path.exists(val_file):
        print(f"[WARNING] Validation file not found: {val_file}")
        print("[INFO] Validation will be skipped")
        return None, None
    
    val_df = pd.read_csv(val_file)
    print(f"Loaded {len(val_df):,} validation samples")
    
    # Check class distribution
    if 'label' in val_df.columns:
        class_dist = val_df['label'].value_counts()
        total = len(val_df)
        print(f"Class distribution (natural imbalance):")
        print(f"  Positive: {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total*100:.2f}%)")
        print(f"  Negative: {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total*100:.2f}%)")
        if class_dist.get(1, 0) > 0:
            ratio = class_dist.get(0, 0) / class_dist.get(1, 0)
            print(f"  Ratio: {ratio:.1f}:1 (Neg:Pos)")
    
    # Select features (only those available in training)
    available_features = [f for f in feature_cols if f in val_df.columns]
    
    if len(available_features) != len(feature_cols):
        print(f"[WARNING] Feature mismatch: {len(available_features)} vs {len(feature_cols)}")
    
    X_val = val_df[available_features].values
    y_val = val_df['label'].values if 'label' in val_df.columns else None
    
    print(f"Validation set shape: {X_val.shape}")
    
    return X_val, y_val

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_baseline_model(X_train, y_train):
    """Train baseline Random Forest model."""
    print("\n" + "=" * 70)
    print("TRAINING BASELINE RANDOM FOREST MODEL")
    print("=" * 70)
    
    print("Configuration:")
    for key, value in RF_CONFIG.items():
        print(f"  {key}: {value}")
    
    # Create model
    rf_model = RandomForestClassifier(**RF_CONFIG)
    
    # Train
    print("\nTraining model...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    # OOB score (if available)
    if hasattr(rf_model, 'oob_score_'):
        print(f"Out-of-bag score: {rf_model.oob_score_:.4f}")
    
    return rf_model, training_time

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X, y, split_name="", feature_cols=None):
    """Evaluate model performance."""
    print("\n" + "=" * 70)
    print(f"EVALUATION - {split_name.upper()}")
    print("=" * 70)
    
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Metrics
    accuracy = (y_pred == y).mean()
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    print(f"Performance Metrics:")
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
    print(f"\nAdditional Metrics:")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")
    
    # Feature importance (if available)
    if feature_cols and hasattr(model, 'feature_importances_'):
        print(f"\nTop 5 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'fpr': float(fpr),
        'fnr': float(fnr)
    }

# =============================================================================
# SAVE MODEL AND METADATA
# =============================================================================

def save_model_and_metadata(model, feature_cols, training_metrics, validation_metrics, training_time):
    """Save trained model and metadata."""
    print("\n" + "=" * 70)
    print("SAVING MODEL AND METADATA")
    print("=" * 70)
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = os.path.join(MODELS_DIR, f"baseline_rf_model_{timestamp}.pkl")
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'model_name': 'baseline_rf_model',
        'hyperparameters': RF_CONFIG,
        'features': feature_cols,
        'n_features': len(feature_cols),
        'feature_set': 'original_15',
        'training_time_seconds': float(training_time),
        'training_metrics': training_metrics,
        'validation_metrics': validation_metrics,
        'oob_score': float(model.oob_score_) if hasattr(model, 'oob_score_') else None
    }
    
    metadata_filename = os.path.join(MODELS_DIR, f"baseline_rf_metadata_{timestamp}.json")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_filename}")
    
    # Save results summary
    results_summary = {
        'timestamp': timestamp,
        'model_name': 'baseline_rf_model',
        'training': training_metrics,
        'validation': validation_metrics,
        'training_time_seconds': float(training_time)
    }
    
    results_filename = os.path.join(RESULTS_DIR, f"baseline_training_results_{timestamp}.json")
    with open(results_filename, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to: {results_filename}")
    
    return model_filename, metadata_filename

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("BASELINE RANDOM FOREST TRAINING - COMPREHENSIVE DATASET")
    print("=" * 70)
    print(f"Phase: 3.1 - Baseline Model (15 original features)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load training data
    X_train, y_train, feature_cols = load_training_data()
    if X_train is None:
        return 1
    
    # Load validation data
    X_val, y_val = load_validation_data(feature_cols)
    
    # Train model
    model, training_time = train_baseline_model(X_train, y_train)
    
    # Evaluate on training set
    training_metrics = evaluate_model(model, X_train, y_train, "Training", feature_cols)
    
    # Evaluate on validation set (if available)
    validation_metrics = None
    if X_val is not None and y_val is not None:
        validation_metrics = evaluate_model(model, X_val, y_val, "Validation", feature_cols)
    
    # Save model and metadata
    model_file, metadata_file = save_model_and_metadata(
        model, feature_cols, training_metrics, validation_metrics, training_time
    )
    
    print("\n" + "=" * 70)
    print("BASELINE MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Review validation metrics")
    print(f"  2. Compare with future models (with autoencoder features)")
    print(f"  3. Proceed to Phase 3.2: Model with Autoencoder Features")
    
    return 0

if __name__ == "__main__":
    exit(main())

