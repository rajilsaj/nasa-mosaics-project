"""
Random Forest Training for Mars Vortex Detection
=================================================

This script trains a Random Forest classifier for on-board vortex detection.
It is aligned with the modular pipeline:
split_data.py -> extract_windows.py -> negative_sampling.py -> feature_engineering.py

Key Features:
- Handles class imbalance with class_weight='balanced'
- Validation-based threshold selection (scientifically correct)
- Test evaluation at frozen validation-selected threshold
- Feature importance analysis
- Model + metadata persistence for deployment
"""

import argparse
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Training configuration."""
    
    # File paths (default assumes feature_engineering.py output in current directory)
    TRAIN_FILE = "train_features.csv"
    VAL_FILE = "val_features.csv"
    TEST_FILE = "test_features.csv"
    
    # Output directory
    OUTPUT_DIR = "models"
    RESULTS_DIR = "results"
    
    # Random Forest parameters
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'class_weight': 'balanced',  # Critical for imbalanced data
        'random_state': 42,
        'n_jobs': -1  # Use all cores
    }
    
    # Feature columns (exclude metadata)
    EXCLUDE_COLUMNS = ['window_id', 'event_sclk', 'label']
    
    # Threshold search
    THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    PRIMARY_METRIC = "f1"  # choices: f1, precision, recall
    MIN_PRECISION = 0.0
    MIN_RECALL = 0.0

    # Random seed
    RANDOM_SEED = 42


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RF with validation-selected threshold."
    )
    parser.add_argument(
        "--features_dir",
        default=".",
        help="Directory containing train/val/test feature CSV files.",
    )
    parser.add_argument("--train_file", default=Config.TRAIN_FILE)
    parser.add_argument("--val_file", default=Config.VAL_FILE)
    parser.add_argument("--test_file", default=Config.TEST_FILE)
    parser.add_argument(
        "--primary_metric",
        choices=["f1", "precision", "recall"],
        default=Config.PRIMARY_METRIC,
        help="Metric used to select validation threshold.",
    )
    parser.add_argument(
        "--threshold_grid",
        default=",".join([str(x) for x in Config.THRESHOLD_GRID]),
        help="Comma-separated thresholds, e.g. 0.1,0.2,0.3",
    )
    parser.add_argument("--min_precision", type=float, default=Config.MIN_PRECISION)
    parser.add_argument("--min_recall", type=float, default=Config.MIN_RECALL)
    return parser.parse_args()

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(args):
    """Load training, validation, and test datasets."""
    print("Loading datasets...")

    train_path = os.path.join(args.features_dir, args.train_file)
    val_path = os.path.join(args.features_dir, args.val_file)
    test_path = os.path.join(args.features_dir, args.test_file)

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Required feature file not found: {p}. "
                "Run split/extract/sampling/feature engineering first."
            )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    return train_df, val_df, test_df


def prepare_features(df, feature_cols=None):
    """
    Prepare features and labels from DataFrame.
    
    Args:
        df: DataFrame with features and label
        
    Returns:
        X: Feature matrix
        y: Label vector
    """
    # Derive from training split once, then enforce same order on val/test
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in Config.EXCLUDE_COLUMNS]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[feature_cols].values
    y = df['label'].values

    return X, y, feature_cols

# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_random_forest(X_train, y_train):
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained RandomForestClassifier
    """
    print("\nTraining Random Forest classifier...")
    print(f"  Parameters: {Config.RF_PARAMS}")
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"  Training class distribution: {class_dist}")
    
    # Train model
    start_time = time.time()
    
    rf_model = RandomForestClassifier(**Config.RF_PARAMS)
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    print(f"  Training completed in {training_time:.2f} seconds")
    print(f"  Number of trees: {rf_model.n_estimators}")
    
    return rf_model, training_time

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X, y, threshold=0.5, split_name="Validation"):
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        split_name: Name of the split (for reporting)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING ON {split_name.upper()} SET")
    print(f"{'='*70}")
    
    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"True class distribution: {class_dist}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nDecision threshold: {threshold:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"True Negative   {cm[0,0]:<20} {cm[0,1]:<20}")
    print(f"True Positive   {cm[1,0]:<20} {cm[1,1]:<20}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))
    
    # Detailed Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='binary', zero_division=0
    )
    
    print(f"\nDetailed Metrics (Positive Class):")
    print(f"  Precision: {precision:.4f} (of predicted positives, how many are correct)")
    print(f"  Recall:    {recall:.4f} (of true positives, how many were detected)")
    print(f"  F1-Score:  {f1:.4f} (harmonic mean of precision and recall)")
    
    # ROC AUC
    try:
        auc = roc_auc_score(y, y_pred_proba)
        print(f"  ROC AUC:   {auc:.4f}")
    except:
        auc = None
        print(f"  ROC AUC:   N/A (insufficient data)")
    
    # False Positive/Negative Analysis
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]
    true_negatives = cm[0, 0]
    
    print(f"\nError Analysis:")
    print(f"  False Positives: {false_positives} (incorrectly predicted as vortex)")
    print(f"  False Negatives: {false_negatives} (missed vortex detections)")
    print(f"  True Positives:  {true_positives} (correctly detected vortices)")
    print(f"  True Negatives:  {true_negatives} (correctly identified non-vortex)")
    
    # Return metrics dictionary
    metrics = {
        'split': split_name,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
    
    return metrics


def parse_threshold_grid(raw_grid):
    """Parse comma-separated threshold values."""
    values = []
    for token in raw_grid.split(","):
        token = token.strip()
        if not token:
            continue
        val = float(token)
        if val <= 0.0 or val >= 1.0:
            raise ValueError(f"Invalid threshold {val}; use values in (0, 1).")
        values.append(val)
    if not values:
        raise ValueError("Threshold grid is empty.")
    return sorted(set(values))


def evaluate_threshold_grid(model, X_val, y_val, thresholds):
    """Evaluate validation performance over threshold grid."""
    proba = model.predict_proba(X_val)[:, 1]
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, pred, average="binary", zero_division=0
        )
        rows.append(
            {
                "threshold": t,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(metrics_df, primary_metric, min_precision, min_recall):
    """Select best threshold on validation with optional constraints."""
    eligible = metrics_df[
        (metrics_df["precision"] >= min_precision) &
        (metrics_df["recall"] >= min_recall)
    ].copy()

    if eligible.empty:
        print(
            "[WARNING] No threshold met constraints. Falling back to unconstrained selection."
        )
        eligible = metrics_df.copy()

    # stable tie-break: maximize primary -> maximize f1 -> lower threshold (recall-friendly)
    eligible = eligible.sort_values(
        by=[primary_metric, "f1", "threshold"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return float(eligible.iloc[0]["threshold"]), eligible.iloc[0].to_dict()

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model, feature_names):
    """
    Analyze and report feature importance.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    print(f"\n{'='*70}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': [importances[i] for i in indices]
    })
    
    print("\nTop 10 Most Important Features:")
    for i in range(min(10, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"  {i+1}. {row['feature']:25s}: {row['importance']:.4f}")
    
    return importance_df

# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, feature_names, training_time, val_metrics, test_metrics, best_threshold, args):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        training_time: Training duration in seconds
        metrics: Dictionary of evaluation metrics
    """
    print(f"\n{'='*70}")
    print("SAVING MODEL")
    print(f"{'='*70}")
    
    # Create output directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(Config.OUTPUT_DIR, f"rf_vortex_detector_{timestamp}.pkl")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'rf_params': Config.RF_PARAMS,
        'training_time_seconds': training_time,
        "primary_metric": args.primary_metric,
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "selected_threshold": best_threshold,
        'validation_metrics': {
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1_score': val_metrics['f1'],
            'roc_auc': val_metrics['auc']
        },
        'test_metrics': {
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1'],
            'roc_auc': test_metrics['auc']
        },
    }

    metadata_path = os.path.join(Config.OUTPUT_DIR, f"model_metadata_{timestamp}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Metadata saved to: {metadata_path}")

    return model_path, metadata_path

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    args = parse_args()
    print("="*70)
    print("RANDOM FOREST TRAINING - MARS VORTEX DETECTION")
    print("="*70)

    # Set random seed
    np.random.seed(Config.RANDOM_SEED)

    # Load data
    train_df, val_df, test_df = load_data(args)

    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df, feature_names)
    X_test, y_test, _ = prepare_features(test_df, feature_names)

    print(f"  Feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}")

    # Train model
    rf_model, training_time = train_random_forest(X_train, y_train)

    thresholds = parse_threshold_grid(args.threshold_grid)
    print(f"\nValidation threshold sweep on {len(thresholds)} thresholds...")
    threshold_metrics_df = evaluate_threshold_grid(rf_model, X_val, y_val, thresholds)
    best_threshold, best_row = select_best_threshold(
        threshold_metrics_df,
        primary_metric=args.primary_metric,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    print(
        f"Selected threshold={best_threshold:.3f} "
        f"(precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f}, f1={best_row['f1']:.4f})"
    )

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    threshold_path = os.path.join(Config.RESULTS_DIR, "validation_threshold_sweep.csv")
    threshold_metrics_df.to_csv(threshold_path, index=False)
    print(f"Validation threshold sweep saved to: {threshold_path}")

    # Evaluate with frozen validation-selected threshold
    val_metrics = evaluate_model(rf_model, X_val, y_val, best_threshold, "Validation")
    test_metrics = evaluate_model(rf_model, X_test, y_test, best_threshold, "Test")

    # Feature importance
    importance_df = analyze_feature_importance(rf_model, feature_names)

    # Save feature importance
    importance_path = os.path.join(Config.RESULTS_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")

    # Save model
    model_path, metadata_path = save_model(
        rf_model,
        feature_names,
        training_time,
        val_metrics,
        test_metrics,
        best_threshold,
        args,
    )

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nModel ready for deployment:")
    print(f"  Model file: {model_path}")
    print(f"  Metadata file: {metadata_path}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Selected threshold: {best_threshold:.3f}")
    print(f"  Validation F1-Score: {val_metrics['f1']:.4f}")
    print(f"  Test F1-Score: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()

