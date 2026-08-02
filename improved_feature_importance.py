#!/usr/bin/env python3
"""
Improved Feature Importance Visualization with Better Label Readability
=====================================================================

This script creates feature importance visualizations with clear, readable labels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
    
    # Prepare X and y
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
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

def create_readable_feature_importance_plot(model, feature_cols, output_path="improved_feature_importance.png"):
    """Create feature importance visualization with readable labels."""
    print(f"\nCreating improved feature importance visualization...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Create the plot with better layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # Horizontal bar plot (top)
    bars = ax1.barh(importance_df['feature'], importance_df['importance'], 
                    color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    
    ax1.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Features', fontsize=14, fontweight='bold')
    ax1.set_title('Random Forest Feature Importance\n(Mars Vortex Detection Training Set)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        ax1.text(importance + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f} ({importance*100:.1f}%)', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Customize appearance
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(importances) * 1.3)
    
    # Improve y-axis labels
    ax1.tick_params(axis='y', labelsize=12)
    
    # Pie chart for top 10 features (bottom) with better label positioning
    top_10 = importance_df.tail(10)
    
    # Create custom colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
              '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
    
    # Create pie chart with better label positioning
    wedges, texts, autotexts = ax2.pie(top_10['importance'], 
                                       labels=top_10['feature'],
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90,
                                       textprops={'fontsize': 10, 'fontweight': 'bold'},
                                       pctdistance=0.85,
                                       labeldistance=1.1)
    
    ax2.set_title('Top 10 Feature Importance Distribution\n(% of Total Importance)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Make labels more readable
    for text in texts:
        text.set_fontsize(10)
        text.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Improved feature importance plot saved to: {output_path}")
    
    return importance_df

def create_alternative_pie_chart(model, feature_cols, output_path="alternative_pie_chart.png"):
    """Create an alternative pie chart with legend instead of labels."""
    print(f"\nCreating alternative pie chart with legend...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Get top 10
    top_10 = importance_df.head(10)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create custom colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', 
              '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
    
    # Create pie chart with legend
    wedges, texts, autotexts = ax.pie(top_10['importance'], 
                                      autopct='%1.1f%%',
                                      colors=colors,
                                      startangle=90,
                                      textprops={'fontsize': 12, 'fontweight': 'bold'})
    
    # Create legend with feature names and percentages
    legend_labels = [f"{feature}: {importance*100:.1f}%" 
                    for feature, importance in zip(top_10['feature'], top_10['importance'])]
    
    ax.legend(wedges, legend_labels, 
              title="Feature Importance",
              loc="center left", 
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)
    
    ax.set_title('Top 10 Feature Importance Distribution\n(% of Total Importance)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Alternative pie chart saved to: {output_path}")

def create_horizontal_bar_chart(model, feature_cols, output_path="horizontal_bar_chart.png"):
    """Create a clean horizontal bar chart for feature importance."""
    print(f"\nCreating horizontal bar chart...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create horizontal bar chart
    bars = ax.barh(importance_df['feature'], importance_df['importance'], 
                   color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    
    ax.set_xlabel('Feature Importance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Features', fontsize=14, fontweight='bold')
    ax.set_title('Random Forest Feature Importance\n(Mars Vortex Detection Training Set)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        ax.text(importance + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f} ({importance*100:.1f}%)', 
                va='center', ha='left', fontsize=11, fontweight='bold')
    
    # Customize appearance
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(importances) * 1.3)
    ax.tick_params(axis='y', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Horizontal bar chart saved to: {output_path}")

def main():
    """Main execution function."""
    print("="*70)
    print("IMPROVED FEATURE IMPORTANCE VISUALIZATION")
    print("="*70)
    
    try:
        # 1. Load training data
        X_train, y_train, feature_cols = load_training_data()
        
        # 2. Train model
        model = train_model(X_train, y_train)
        
        # 3. Create improved visualizations
        importance_df = create_readable_feature_importance_plot(model, feature_cols)
        create_alternative_pie_chart(model, feature_cols)
        create_horizontal_bar_chart(model, feature_cols)
        
        print(f"\n{'='*70}")
        print("IMPROVED VISUALIZATIONS COMPLETED")
        print(f"{'='*70}")
        print("Generated visualizations:")
        print("  - improved_feature_importance.png (combined plot)")
        print("  - alternative_pie_chart.png (pie chart with legend)")
        print("  - horizontal_bar_chart.png (clean bar chart)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()



