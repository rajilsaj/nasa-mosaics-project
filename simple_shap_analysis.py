#!/usr/bin/env python3
"""
Simple SHAP Analysis for Mars Vortex Detection Training Set
===========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import SHAP
import shap

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
    X_train = train_features[feature_cols].values  # Convert to numpy array
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

def create_shap_analysis(rf_model, X_train, feature_cols, output_path="shap_summary_training.png"):
    """Create SHAP summary plot for model interpretability."""
    print(f"\nCreating SHAP analysis...")
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_model)
        
        # Calculate SHAP values for a subset of data (faster)
        n_samples = min(100, len(X_train))  # Use first 100 samples for speed
        X_sample = X_train[:n_samples]
        
        print(f"  Calculating SHAP values for {n_samples} samples...")
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, use the positive class (index 1)
        if len(shap_values) == 2:
            shap_values_pos = shap_values[1]
        else:
            shap_values_pos = shap_values
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_pos, X_sample, feature_names=feature_cols, 
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
        shap.summary_plot(shap_values_pos, X_sample, feature_names=feature_cols, 
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
        print(f"  Error details: {str(e)}")
        return None

def main():
    """Main execution function."""
    print("="*70)
    print("SHAP ANALYSIS - TRAINING SET (SIMPLE)")
    print("="*70)
    
    try:
        # 1. Load training data
        X_train, y_train, feature_cols = load_training_data()
        
        # 2. Train model
        rf_model = train_model(X_train, y_train)
        
        # 3. Create SHAP analysis
        shap_importance_df = create_shap_analysis(rf_model, X_train, feature_cols)
        
        if shap_importance_df is not None:
            print(f"\n{'='*70}")
            print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"{'='*70}")
            print("Generated visualizations:")
            print("  - shap_summary_training.png")
            print("  - shap_summary_training_bar.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()



