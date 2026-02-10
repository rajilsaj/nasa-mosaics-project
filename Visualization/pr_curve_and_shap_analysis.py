#!/usr/bin/env python3
"""
PR Curve and SHAP Analysis for Mars Vortex Detection Training Set
===============================================================

This script creates:
1. Precision-Recall curve for training set
2. SHAP summary plot for model interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, install if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    SHAP_AVAILABLE = True

def load_training_data():
    """Load training features and prepare data."""
    print("Loading training data...")
    
    # Load training features
    train_features = pd.read_csv('train_features.csv')
    print(f"  Loaded {len(train_features):,} training samples")
    
    # Prepare feature columns
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    
    # Prepare X and y
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
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

def create_pr_curve(rf_model, X_train, y_train, output_path="pr_curve_training.png"):
    """Create Precision-Recall curve for training set."""
    print(f"\nCreating Precision-Recall curve...")
    
    # Get prediction probabilities
    y_proba = rf_model.predict_proba(X_train)[:, 1]
    
    # Calculate PR curve
    precision, recall, thresholds = precision_recall_curve(y_train, y_proba)
    avg_precision = average_precision_score(y_train, y_proba)
    
    # Calculate ROC curve for comparison
    fpr, tpr, roc_thresholds = roc_curve(y_train, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PR Curve
    ax1.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
    ax1.axhline(y=np.mean(y_train), color='r', linestyle='--', alpha=0.7, 
                label=f'Random Classifier (AP = {np.mean(y_train):.3f})')
    ax1.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision-Recall Curve\n(Training Set)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # ROC Curve
    ax2.plot(fpr, tpr, 'g-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Random Classifier (AUC = 0.500)')
    ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax2.set_title('ROC Curve\n(Training Set)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  PR and ROC curves saved to: {output_path}")
    
    # Print summary statistics
    print(f"\nPR Curve Summary:")
    print(f"  Average Precision: {avg_precision:.4f}")
    print(f"  Random Classifier AP: {np.mean(y_train):.4f}")
    print(f"  Improvement: {avg_precision / np.mean(y_train):.2f}x better than random")
    
    print(f"\nROC Curve Summary:")
    print(f"  AUC: {roc_auc:.4f}")
    print(f"  Perfect classifier AUC: 1.000")
    print(f"  Random classifier AUC: 0.500")
    
    return avg_precision, roc_auc

def create_shap_analysis(rf_model, X_train, feature_cols, output_path="shap_summary_training.png"):
    """Create SHAP summary plot for model interpretability."""
    print(f"\nCreating SHAP analysis...")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_train)
        
        # For binary classification, use the positive class (index 1)
        if len(shap_values) == 2:
            shap_values_pos = shap_values[1]
        else:
            shap_values_pos = shap_values
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_pos, X_train, feature_names=feature_cols, 
                         show=False, max_display=15)
        plt.title('SHAP Summary Plot - Training Set\n(Feature Impact on Precursor Detection)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  SHAP summary plot saved to: {output_path}")
        
        # Calculate and print feature importance from SHAP
        feature_importance = np.abs(shap_values_pos).mean(0)
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': feature_importance
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\nSHAP Feature Importance (Top 10):")
        print("-" * 50)
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<20} {row['shap_importance']:.4f}")
        
        # Create SHAP bar plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values_pos, X_train, feature_names=feature_cols, 
                         plot_type="bar", show=False, max_display=15)
        plt.title('SHAP Feature Importance - Training Set\n(Mean Absolute SHAP Values)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        bar_path = output_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"  SHAP bar plot saved to: {bar_path}")
        
        return feature_importance_df
        
    except Exception as e:
        print(f"  Error creating SHAP analysis: {e}")
        return None

def create_threshold_analysis(rf_model, X_train, y_train, output_path="threshold_analysis_training.png"):
    """Create threshold analysis showing precision/recall trade-offs."""
    print(f"\nCreating threshold analysis...")
    
    # Get prediction probabilities
    y_proba = rf_model.predict_proba(X_train)[:, 1]
    
    # Calculate precision, recall, and F1 for different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_train == 1))
        fp = np.sum((y_pred == 1) & (y_train == 0))
        fn = np.sum((y_pred == 0) & (y_train == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision', marker='o')
    plt.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall', marker='s')
    plt.plot(thresholds, f1_scores, 'g-', linewidth=2, label='F1-Score', marker='^')
    
    plt.xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Threshold Analysis - Training Set\n(Precision vs Recall Trade-offs)', 
             fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim([0.1, 0.95])
    plt.ylim([0, 1])
    
    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    optimal_precision = precisions[optimal_idx]
    optimal_recall = recalls[optimal_idx]
    
    plt.axvline(x=optimal_threshold, color='purple', linestyle='--', alpha=0.7,
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.axhline(y=optimal_f1, color='purple', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Threshold analysis saved to: {output_path}")
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Optimal F1-Score: {optimal_f1:.4f}")
    print(f"  Optimal Precision: {optimal_precision:.4f}")
    print(f"  Optimal Recall: {optimal_recall:.4f}")
    
    return optimal_threshold, optimal_f1

def main():
    """Main execution function."""
    print("="*70)
    print("PR CURVE AND SHAP ANALYSIS - TRAINING SET")
    print("="*70)
    
    try:
        # 1. Load training data
        X_train, y_train, feature_cols = load_training_data()
        
        # 2. Train model
        rf_model = train_model(X_train, y_train)
        
        # 3. Create PR curve
        avg_precision, roc_auc = create_pr_curve(rf_model, X_train, y_train)
        
        # 4. Create SHAP analysis
        shap_importance_df = create_shap_analysis(rf_model, X_train, feature_cols)
        
        # 5. Create threshold analysis
        optimal_threshold, optimal_f1 = create_threshold_analysis(rf_model, X_train, y_train)
        
        print(f"\n{'='*70}")
        print("PR CURVE AND SHAP ANALYSIS COMPLETED")
        print(f"{'='*70}")
        print("Generated visualizations:")
        print("  - pr_curve_training.png")
        print("  - shap_summary_training.png")
        print("  - shap_summary_training_bar.png")
        print("  - threshold_analysis_training.png")
        
        print(f"\nSummary:")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Optimal Threshold: {optimal_threshold:.3f}")
        print(f"  Optimal F1-Score: {optimal_f1:.4f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()



