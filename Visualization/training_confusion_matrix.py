#!/usr/bin/env python3
"""
Training Set Confusion Matrix Analysis
=====================================

This script creates a confusion matrix for the training set to analyze
model performance on the data it was trained on.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Load training features and prepare data."""
    print("Loading training data...")
    
    # Load training features
    train_features = pd.read_csv('datasets/train_features.csv')
    print(f"  Loaded {len(train_features):,} training samples")
    
    # Prepare feature columns
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature columns: {feature_cols}")
    
    # Prepare X and y
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    print(f"  Class balance: {np.bincount(y_train) / len(y_train) * 100}")
    
    return X_train, y_train, feature_cols

def train_model(X_train, y_train):
    """Train Random Forest model on training data."""
    print("\nTraining Random Forest model...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    print("  Model training completed!")
    
    return rf_model

def create_confusion_matrix_plot(y_true, y_pred, model_name="Random Forest"):
    """Create and save confusion matrix visualization."""
    print(f"\nCreating confusion matrix for {model_name}...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confusion matrix (raw counts)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Precursor', 'Precursor'], 
                yticklabels=['No Precursor', 'Precursor'],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'Confusion Matrix - Raw Counts\n{model_name} on Training Set')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Add performance metrics to the plot
    ax1.text(0.5, -0.15, f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
             transform=ax1.transAxes, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Confusion matrix (normalized)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=['No Precursor', 'Precursor'], 
                yticklabels=['No Precursor', 'Precursor'],
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title(f'Confusion Matrix - Normalized\n{model_name} on Training Set')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # Add detailed breakdown
    tn, fp, fn, tp = cm.ravel()
    breakdown_text = f'TN: {tn} | FP: {fp}\nFN: {fn} | TP: {tp}'
    ax2.text(0.5, -0.15, breakdown_text, 
             transform=ax2.transAxes, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "training_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Confusion matrix saved to: {output_file}")
    
    return cm, accuracy, precision, recall, f1

def analyze_training_performance(X_train, y_train, model):
    """Analyze detailed training performance."""
    print("\n" + "="*60)
    print("TRAINING SET PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Get predictions
    y_pred = model.predict(X_train)
    y_proba = model.predict_proba(X_train)[:, 1]
    
    # Create confusion matrix
    cm, accuracy, precision, recall, f1 = create_confusion_matrix_plot(y_train, y_pred)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_train, y_pred, 
                              target_names=['No Precursor', 'Precursor'],
                              digits=4))
    
    # Feature importance analysis
    print("\nTop 10 Most Important Features:")
    feature_importance = model.feature_importances_
    feature_names = [col for col in pd.read_csv('datasets/train_features.csv').columns 
                    if col not in ['window_id', 'event_sclk', 'label']]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Training set characteristics
    print(f"\nTraining Set Characteristics:")
    print(f"  Total samples: {len(y_train):,}")
    print(f"  Positive samples: {np.sum(y_train):,} ({np.mean(y_train)*100:.1f}%)")
    print(f"  Negative samples: {len(y_train) - np.sum(y_train):,} ({100-np.mean(y_train)*100:.1f}%)")
    print(f"  Model complexity: {model.n_estimators} trees, max_depth={model.max_depth}")
    
    # Overfitting check
    print(f"\nOverfitting Analysis:")
    print(f"  Training accuracy: {accuracy:.4f}")
    print(f"  Training F1-score: {f1:.4f}")
    if accuracy > 0.99:
        print("  [WARNING] Very high training accuracy may indicate overfitting!")
    elif accuracy > 0.95:
        print("  [CAUTION] High training accuracy - monitor validation performance")
    else:
        print("  [OK] Training accuracy appears reasonable")
    
    return cm, accuracy, precision, recall, f1

def main():
    """Main execution function."""
    print("="*60)
    print("TRAINING SET CONFUSION MATRIX ANALYSIS")
    print("="*60)
    
    try:
        # 1. Load training data
        X_train, y_train, feature_cols = load_training_data()
        
        # 2. Train model
        model = train_model(X_train, y_train)
        
        # 3. Analyze training performance
        cm, accuracy, precision, recall, f1 = analyze_training_performance(X_train, y_train, model)
        
        print(f"\n{'='*60}")
        print("TRAINING ANALYSIS COMPLETED")
        print(f"{'='*60}")
        print(f"Training Performance Summary:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"\nConfusion matrix visualization saved to: training_confusion_matrix.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
