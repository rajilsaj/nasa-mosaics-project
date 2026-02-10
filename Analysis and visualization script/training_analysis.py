#!/usr/bin/env python3
"""
Comprehensive Training Set Analysis for Random Forest Model
Creates multiple visualizations to understand training data characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_training_data():
    """Load and prepare training features."""
    print("Loading training features...")
    df = pd.read_csv('train_features.csv')
    
    print(f"Training data shape: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['window_id', 'event_sclk', 'label']]
    X = df[feature_cols]
    y = df['label']
    
    return df, X, y, feature_cols

def create_twinx_feature_plots(X, y, feature_cols, save_path='training_analysis_twinx.png'):
    """Create twinx plots showing feature distributions for both classes."""
    print("\nCreating twinx feature distribution plots...")
    
    # Calculate number of subplots needed
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        
        # Get data for each class
        pos_data = X[y == 1][feature]
        neg_data = X[y == 0][feature]
        
        # Create histogram for negative class
        ax.hist(neg_data, bins=30, alpha=0.7, color='skyblue', label='Non-Vortex', density=True)
        ax.set_xlabel(f'{feature}')
        ax.set_ylabel('Density (Non-Vortex)', color='skyblue')
        ax.tick_params(axis='y', labelcolor='skyblue')
        
        # Create twin axis for positive class
        ax2 = ax.twinx()
        ax2.hist(pos_data, bins=30, alpha=0.7, color='red', label='Vortex', density=True)
        ax2.set_ylabel('Density (Vortex)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add title and statistics
        pos_mean, pos_std = pos_data.mean(), pos_data.std()
        neg_mean, neg_std = neg_data.mean(), neg_data.std()
        ax.set_title(f'{feature}\nVortex: {pos_mean:.3f}±{pos_std:.3f}\nNon-Vortex: {neg_mean:.3f}±{neg_std:.3f}')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Twinx plots saved to: {save_path}")
    plt.show()

def create_feature_importance_plot(X, y, feature_cols, save_path='training_analysis_importance.png'):
    """Create feature importance plot from Random Forest."""
    print("\nCreating feature importance plot...")
    
    # Train a quick RF to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance\n(Trained on Training Set)')
    plt.grid(axis='x', alpha=0.3)
    
    # Color bars by importance
    colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add value labels
    for i, (feature, importance) in enumerate(zip(importance_df['feature'], importance_df['importance'])):
        plt.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.show()
    
    return importance_df

def create_correlation_heatmap(X, save_path='training_analysis_correlation.png'):
    """Create correlation heatmap of features."""
    print("\nCreating correlation heatmap...")
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = X.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
    
    plt.title('Feature Correlation Heatmap\n(Training Set)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {save_path}")
    plt.show()

def create_box_plots(X, y, feature_cols, save_path='training_analysis_boxplots.png'):
    """Create box plots for class separation analysis."""
    print("\nCreating box plots for class separation...")
    
    # Select top 6 most important features for box plots
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    top_features = np.array(feature_cols)[np.argsort(rf.feature_importances_)[-6:]]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        # Create box plot
        data_to_plot = [X[y == 0][feature], X[y == 1][feature]]
        bp = axes[i].boxplot(data_to_plot, labels=['Non-Vortex', 'Vortex'], patch_artist=True)
        
        # Color the boxes
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        
        axes[i].set_title(f'{feature}')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Box Plots: Class Separation Analysis\n(Top 6 Most Important Features)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Box plots saved to: {save_path}")
    plt.show()

def create_statistical_summary(df, feature_cols, save_path='training_analysis_summary.txt'):
    """Create statistical summary of training data."""
    print("\nCreating statistical summary...")
    
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TRAINING SET STATISTICAL ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Samples: {len(df)}\n\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        f.write(f"Positive (Vortex): {len(df[df['label'] == 1])} ({len(df[df['label'] == 1])/len(df)*100:.1f}%)\n")
        f.write(f"Negative (Non-Vortex): {len(df[df['label'] == 0])} ({len(df[df['label'] == 0])/len(df)*100:.1f}%)\n\n")
        
        f.write("FEATURE STATISTICS:\n")
        f.write("-"*50 + "\n")
        for feature in feature_cols:
            pos_data = df[df['label'] == 1][feature]
            neg_data = df[df['label'] == 0][feature]
            
            f.write(f"\n{feature}:\n")
            f.write(f"  Vortex:     mean={pos_data.mean():.4f}, std={pos_data.std():.4f}, range=[{pos_data.min():.4f}, {pos_data.max():.4f}]\n")
            f.write(f"  Non-Vortex: mean={neg_data.mean():.4f}, std={neg_data.std():.4f}, range=[{neg_data.min():.4f}, {neg_data.max():.4f}]\n")
            
            # Calculate separation
            separation = abs(pos_data.mean() - neg_data.mean()) / (pos_data.std() + neg_data.std())
            f.write(f"  Separation: {separation:.4f}\n")
    
    print(f"Statistical summary saved to: {save_path}")

def main():
    """Main analysis function."""
    print("="*70)
    print("COMPREHENSIVE TRAINING SET ANALYSIS")
    print("="*70)
    print("Analyzing Random Forest training data characteristics")
    print("="*70)
    
    # Load data
    df, X, y, feature_cols = load_training_data()
    
    # Create all visualizations
    create_twinx_feature_plots(X, y, feature_cols)
    importance_df = create_feature_importance_plot(X, y, feature_cols)
    create_correlation_heatmap(X)
    create_box_plots(X, y, feature_cols)
    create_statistical_summary(df, feature_cols)
    
    print(f"\n{'='*70}")
    print("TRAINING ANALYSIS COMPLETED")
    print(f"{'='*70}")
    print("Generated files:")
    print("  📊 training_analysis_twinx.png - Twinx feature distributions")
    print("  📊 training_analysis_importance.png - Feature importance")
    print("  📊 training_analysis_correlation.png - Feature correlations")
    print("  📊 training_analysis_boxplots.png - Class separation")
    print("  📋 training_analysis_summary.txt - Statistical summary")
    print("\nThis analysis helps understand:")
    print("  • How well features separate classes (twinx plots)")
    print("  • Which features RF considers most important")
    print("  • Feature relationships and correlations")
    print("  • Data quality and distributions")
    print("="*70)

if __name__ == "__main__":
    main()


