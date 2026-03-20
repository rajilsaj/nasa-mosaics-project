"""
Random Forest Training for Mars Vortex Detection
=================================================

This script trains a Random Forest classifier for on-board vortex detection.
Designed for deployment on Qualcomm Snapdragon-class processors.

Key Features:
- Handles class imbalance with class_weight='balanced'
- Hyperparameter tuning
- Comprehensive evaluation metrics (Precision, Recall, F1-score)
- Feature importance analysis
- Model persistence for deployment
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import time
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Training configuration."""
    
    # File paths
    #TRAIN_FILE = "train_features.csv"
    #VAL_FILE = "val_features.csv"
    TEST_FILE = "test_features.csv"
    
    # Output directory
    #OUTPUT_DIR = "models"
    #RESULTS_DIR = "results"
    
    
    # Base path to your Drive folder
    BASE_PATH = '/content/drive/MyDrive/2026/www/raw/'

    # File paths
    TRAIN_FILE = os.path.join(BASE_PATH, "train_features.csv")
    VAL_FILE   = os.path.join(BASE_PATH, "val_features.csv")
    TEST_FILE  = os.path.join(BASE_PATH, "test_features.csv")

    # Output directories
    OUTPUT_DIR  = os.path.join(BASE_PATH, "models")
    RESULTS_DIR = os.path.join(BASE_PATH, "results")

    # Create output dirs if they don't exist
    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Verify input files exist
    for f in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
        status = "✅ Found" if os.path.exists(f) else "❌ NOT FOUND"
        print(f"{status}: {f}")
    
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
    
    # Random seed
    RANDOM_SEED = 42

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load training, validation, and test datasets."""
    print("Loading datasets...")
    
    train_df = pd.read_csv(Config.TRAIN_FILE)
    val_df = pd.read_csv(Config.VAL_FILE)
    test_df = pd.read_csv(Config.TEST_FILE)
    
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def prepare_features(df):
    """
    Prepare features and labels from DataFrame.
    
    Args:
        df: DataFrame with features and label
        
    Returns:
        X: Feature matrix
        y: Label vector
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in Config.EXCLUDE_COLUMNS]
    
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

def evaluate_model(model, X, y, split_name="Validation"):
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
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"True class distribution: {class_dist}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"True Negative   {cm[0,0]:<20} {cm[0,1]:<20}")
    print(f"True Positive   {cm[1,0]:<20} {cm[1,1]:<20}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Negative', 'Positive']))
    
    # Detailed Metrics
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='binary')
    
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

def save_model(model, feature_names, training_time, metrics):
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
        'validation_metrics': {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'roc_auc': metrics['auc']
        }
    }
    
    metadata_path = os.path.join(Config.OUTPUT_DIR, f"model_metadata_{timestamp}.txt")
    with open(metadata_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"  Metadata saved to: {metadata_path}")
    
    return model_path

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    print("="*70)
    print("RANDOM FOREST TRAINING - MARS VORTEX DETECTION")
    print("="*70)
    
    # Set random seed
    np.random.seed(Config.RANDOM_SEED)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Prepare features
    print("\nPreparing features...")
    X_train, y_train, feature_names = prepare_features(train_df)
    X_val, y_val, _ = prepare_features(val_df)
    X_test, y_test, _ = prepare_features(test_df)
    
    print(f"  Feature count: {len(feature_names)}")
    print(f"  Features: {feature_names}")
    
    # Train model
    rf_model, training_time = train_random_forest(X_train, y_train)
    
    # Evaluate on validation set
    val_metrics = evaluate_model(rf_model, X_val, y_val, "Validation")
    
    # Evaluate on test set (final evaluation)
    test_metrics = evaluate_model(rf_model, X_test, y_test, "Test")
    
    # Feature importance
    importance_df = analyze_feature_importance(rf_model, feature_names)
    
    # Save feature importance
    importance_path = os.path.join(Config.RESULTS_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\nFeature importance saved to: {importance_path}")
    
    # Save model
    model_path = save_model(rf_model, feature_names, training_time, val_metrics)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nModel ready for deployment:")
    print(f"  Model file: {model_path}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Validation F1-Score: {val_metrics['f1']:.4f}")
    print(f"  Test F1-Score: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()

