#!/usr/bin/env python3
"""
Feature Importance Analysis for Mars Vortex Detection Training Set
================================================================

This script creates comprehensive feature importance visualizations
for the Random Forest model trained on the Mars vortex detection dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def load_training_data():
    """Load training features and prepare data."""
    print("Loading training data...")
    
    # Load training features
    train_features = pd.read_csv('train_features.csv')
    print(f"  Loaded {len(train_features):,} training samples")
    
    # Prepare feature columns
    feature_cols = [col for col in train_features.columns 
                   if col not in ['window_id', 'event_sclk', 'label']]
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution: {np.bincount(train_features['label'].values)}")
    
    # Prepare X and y
    X_train = train_features[feature_cols].values
    y_train = train_features['label'].values
    
    return X_train, y_train, feature_cols, train_features

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

def create_feature_importance_plot(model, feature_cols, output_path="feature_importance.png"):
    """Create comprehensive feature importance visualization."""
    print(f"\nCreating feature importance visualization...")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Horizontal bar plot
    bars = ax1.barh(importance_df['feature'], importance_df['importance'], 
                    color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax1.set_title('Random Forest Feature Importance\n(Mars Vortex Detection Training Set)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        ax1.text(importance + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', va='center', ha='left', fontsize=9)
    
    # Customize appearance
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(importances) * 1.2)
    
    # Pie chart for top 10 features
    top_10 = importance_df.tail(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_10)))
    
    wedges, texts, autotexts = ax2.pie(top_10['importance'], 
                                       labels=top_10['feature'],
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90,
                                       textprops={'fontsize': 9})
    
    ax2.set_title('Top 10 Feature Importance Distribution\n(% of Total Importance)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Feature importance plot saved to: {output_path}")
    
    return importance_df

def create_detailed_importance_analysis(importance_df, output_path="detailed_feature_analysis.png"):
    """Create detailed feature analysis with categories."""
    print(f"\nCreating detailed feature analysis...")
    
    # Categorize features
    feature_categories = {
        'Temporal Features': ['overall_slope', 'first_half_slope', 'second_half_slope', 'trend_consistency'],
        'Pressure Dynamics': ['pressure_drop', 'drop_rate', 'min_position'],
        'Statistical Measures': ['mean', 'std', 'range', 'first_half_mean', 'second_half_mean', 'mean_ratio'],
        'Anomaly Detection': ['min_zscore', 'anomaly_strength']
    }
    
    # Create category importance
    category_importance = {}
    for category, features in feature_categories.items():
        category_importance[category] = importance_df[
            importance_df['feature'].isin(features)
        ]['importance'].sum()
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Category importance bar chart
    categories = list(category_importance.keys())
    cat_importances = list(category_importance.values())
    
    bars1 = ax1.bar(categories, cat_importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('Feature Category Importance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total Importance', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, importance in zip(bars1, cat_importances):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{importance:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Top 5 features detailed
    top_5 = importance_df.tail(5)
    bars2 = ax2.barh(range(len(top_5)), top_5['importance'], color='steelblue', alpha=0.8)
    ax2.set_yticks(range(len(top_5)))
    ax2.set_yticklabels(top_5['feature'], fontsize=10)
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('Top 5 Most Important Features', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars2, top_5['importance'])):
        ax2.text(importance + 0.005, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', ha='left', fontsize=9)
    
    # 3. Cumulative importance
    cumulative_importance = np.cumsum(importance_df['importance'].values[::-1])
    ax3.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             marker='o', linewidth=2, markersize=6, color='red')
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% threshold')
    ax3.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
    ax3.set_xlabel('Number of Features', fontsize=12)
    ax3.set_ylabel('Cumulative Importance', fontsize=12)
    ax3.set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Feature importance distribution
    ax4.hist(importance_df['importance'], bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.axvline(importance_df['importance'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {importance_df["importance"].mean():.4f}')
    ax4.axvline(importance_df['importance'].median(), color='orange', linestyle='--', 
                linewidth=2, label=f'Median: {importance_df["importance"].median():.4f}')
    ax4.set_xlabel('Feature Importance', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Detailed feature analysis saved to: {output_path}")
    
    return category_importance

def print_feature_importance_summary(importance_df, category_importance):
    """Print comprehensive feature importance summary."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\nTop 10 Most Important Features:")
    print("-" * 50)
    for i, (_, row) in enumerate(importance_df.tail(10).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<20} {row['importance']:.4f} ({row['importance']*100:.1f}%)")
    
    print(f"\nFeature Category Analysis:")
    print("-" * 50)
    total_importance = sum(category_importance.values())
    for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
        percentage = (importance / total_importance) * 100
        print(f"{category:<20} {importance:.4f} ({percentage:.1f}%)")
    
    print(f"\nStatistical Summary:")
    print("-" * 50)
    print(f"Total Features: {len(importance_df)}")
    print(f"Mean Importance: {importance_df['importance'].mean():.4f}")
    print(f"Median Importance: {importance_df['importance'].median():.4f}")
    print(f"Std Importance: {importance_df['importance'].std():.4f}")
    print(f"Max Importance: {importance_df['importance'].max():.4f}")
    print(f"Min Importance: {importance_df['importance'].min():.4f}")
    
    # Top 5 features account for how much importance?
    top_5_importance = importance_df.tail(5)['importance'].sum()
    print(f"\nTop 5 Features Account for: {top_5_importance:.4f} ({top_5_importance*100:.1f}%) of total importance")
    
    # How many features needed for 80% and 90% importance?
    cumulative = np.cumsum(importance_df['importance'].values[::-1])
    features_80 = np.argmax(cumulative >= 0.8) + 1
    features_90 = np.argmax(cumulative >= 0.9) + 1
    
    print(f"Features needed for 80% importance: {features_80}")
    print(f"Features needed for 90% importance: {features_90}")

def main():
    """Main execution function."""
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS - MARS VORTEX DETECTION")
    print("="*70)
    
    try:
        # 1. Load training data
        X_train, y_train, feature_cols, train_df = load_training_data()
        
        # 2. Train model
        model = train_model(X_train, y_train)
        
        # 3. Create feature importance plot
        importance_df = create_feature_importance_plot(model, feature_cols)
        
        # 4. Create detailed analysis
        category_importance = create_detailed_importance_analysis(importance_df)
        
        # 5. Print summary
        print_feature_importance_summary(importance_df, category_importance)
        
        print(f"\n{'='*70}")
        print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
        print(f"{'='*70}")
        print("Generated visualizations:")
        print("  - feature_importance.png")
        print("  - detailed_feature_analysis.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()



