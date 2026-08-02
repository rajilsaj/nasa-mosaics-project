#!/usr/bin/env python3
"""
Improved Random Forest Training for Extreme Class Imbalance
Addresses 99.4% negative class distribution with optimized class weights
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from datetime import datetime

def load_training_data():
    """Load and prepare training data."""
    print("=" * 70)
    print("IMPROVED RANDOM FOREST TRAINING - EXTREME IMBALANCE HANDLING")
    print("=" * 70)
    
    # Load training features
    train_df = pd.read_csv("datasets/train_features.csv")
    print(f"Loaded {len(train_df)} training samples")
    
    # Prepare features and labels (exclude event_sclk to prevent data leakage)
    feature_cols = [col for col in train_df.columns if col not in ['window_id', 'label', 'event_sclk']]
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print(f"Class ratio: {np.bincount(y_train)[0] / np.bincount(y_train)[1]:.1f}:1 (Negative:Positive)")
    
    return X_train, y_train, feature_cols, train_df

def train_improved_model(X_train, y_train, feature_cols):
    """Train Random Forest with improved class weights for extreme imbalance."""
    print("\n" + "=" * 70)
    print("TRAINING IMPROVED RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Calculate class weights for extreme imbalance
    # For 99.4% negative data, we need to heavily weight the positive class
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    
    # Calculate inverse frequency weights
    weight_negative = total_samples / (2 * class_counts[0])  # Normal weight
    weight_positive = total_samples / (2 * class_counts[1])  # Heavy weight
    
    print(f"Class counts: {class_counts}")
    print(f"Calculated weights: Negative={weight_negative:.2f}, Positive={weight_positive:.2f}")
    
    # Create improved Random Forest with heavy positive class weighting
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,  # Slightly reduced to prevent overfitting
        min_samples_split=15,  # Increased for robustness
        min_samples_leaf=8,   # Increased for robustness
        max_features='sqrt',
        class_weight={0: weight_negative, 1: weight_positive},  # Custom weights
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("Training model with improved class weights...")
    rf_model.fit(X_train, y_train)
    print("Training completed!")
    
    return rf_model

def evaluate_training_performance(X_train, y_train, model, feature_cols):
    """Evaluate model performance on training data."""
    print("\n" + "=" * 70)
    print("TRAINING PERFORMANCE EVALUATION")
    print("=" * 70)
    
    # Make predictions
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    
    # Calculate metrics
    accuracy = (y_pred == y_train).mean()
    precision = confusion_matrix(y_train, y_pred)[1, 1] / (confusion_matrix(y_train, y_pred)[1, 1] + confusion_matrix(y_train, y_pred)[0, 1])
    recall = confusion_matrix(y_train, y_pred)[1, 1] / (confusion_matrix(y_train, y_pred)[1, 1] + confusion_matrix(y_train, y_pred)[1, 0])
    f1 = 2 * (precision * recall) / (precision + recall)
    roc_auc = roc_auc_score(y_train, y_proba)
    
    print(f"Training Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_train, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"Actual    Negative  Positive")
    print(f"Negative  {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"Positive  {cm[1,0]:8d}  {cm[1,1]:8d}")
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 5 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head().iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }

def save_model_and_metadata(model, feature_cols, training_metrics):
    """Save trained model and metadata."""
    print("\n" + "=" * 70)
    print("SAVING MODEL AND METADATA")
    print("=" * 70)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"models/improved_rf_vortex_detector_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    print(f"Model saved to: {model_filename}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'n_estimators': 100,
        'max_depth': 12,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'max_features': 'sqrt',
        'class_weight': 'custom_imbalanced',
        'features': feature_cols,
        'training_samples': len(training_metrics['confusion_matrix'].flatten()),
        'training_metrics': {
            'accuracy': float(training_metrics['accuracy']),
            'precision': float(training_metrics['precision']),
            'recall': float(training_metrics['recall']),
            'f1_score': float(training_metrics['f1_score']),
            'roc_auc': float(training_metrics['roc_auc'])
        }
    }
    
    metadata_filename = f"models/improved_rf_metadata_{timestamp}.json"
    import json
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_filename}")
    
    return model_filename, metadata_filename

def main():
    """Main execution function."""
    print("Starting improved Random Forest training for extreme class imbalance...")
    
    # Step 1: Load training data
    X_train, y_train, feature_cols, train_df = load_training_data()
    
    # Step 2: Train improved model
    rf_model = train_improved_model(X_train, y_train, feature_cols)
    
    # Step 3: Evaluate training performance
    training_metrics = evaluate_training_performance(X_train, y_train, rf_model, feature_cols)
    
    # Step 4: Save model and metadata
    model_filename, metadata_filename = save_model_and_metadata(rf_model, feature_cols, training_metrics)
    
    print("\n" + "=" * 70)
    print("IMPROVED TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Next step: Run threshold tuning on validation set")
    print(f"Model ready for: python improved_threshold_tuning.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
