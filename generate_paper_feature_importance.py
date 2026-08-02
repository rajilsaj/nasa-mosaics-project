#!/usr/bin/env python3
"""
Generate Publication-Quality Feature Importance Graph
========================================================
Creates a professional feature importance visualization for conference paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Publication settings
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11
DPI = 300
FIG_SIZE = (10, 7)

def load_model_and_features():
    """
    Load feature importance for ML dataset Random Forest model.
    
    The ML dataset model uses exactly 15 features:
    1. Trend Features (4): overall_slope, first_half_slope, second_half_slope, trend_consistency
    2. Pressure Drop Features (3): pressure_drop, drop_rate, min_position
    3. Statistical Features (3): mean, std, range
    4. Temporal Evolution (3): first_half_mean, second_half_mean, mean_ratio
    5. Anomaly Features (2): min_zscore, anomaly_strength
    """
    print("Loading feature importance data for ML dataset model...")
    
    # Get script directory
    script_dir = Path(__file__).parent.absolute()
    
    # Expected 15 features from ML dataset model (in canonical order)
    EXPECTED_ML_FEATURES = [
        'overall_slope', 'first_half_slope', 'second_half_slope', 'trend_consistency',
        'pressure_drop', 'drop_rate', 'min_position',
        'mean', 'std', 'range',
        'first_half_mean', 'second_half_mean', 'mean_ratio',
        'min_zscore', 'anomaly_strength'
    ]
    
    # Load feature importance from CSV (preferred method)
    # Try multiple possible paths
    possible_paths = [
        script_dir / "results" / "feature_importance.csv",
        script_dir / "feature_importance.csv",
        Path("results/feature_importance.csv"),
        Path("feature_importance.csv")
    ]
    
    for importance_file in possible_paths:
        if importance_file.exists():
            importance_df = pd.read_csv(str(importance_file))
            print(f"  Loaded feature importance from: {importance_file}")
            
            # Verify all expected features are present
            actual_features = set(importance_df['feature'].tolist())
            expected_features = set(EXPECTED_ML_FEATURES)
            
            if actual_features == expected_features:
                print(f"  [OK] Verified: All 15 ML dataset features present")
                # Sort by importance descending for consistency
                importance_df = importance_df.sort_values('importance', ascending=False)
                return None, importance_df
            else:
                missing = expected_features - actual_features
                extra = actual_features - expected_features
                if missing:
                    print(f"  [WARNING] Missing expected features: {missing}")
                if extra:
                    print(f"  [WARNING] Extra features found: {extra}")
                # Still return it, but warn user
                importance_df = importance_df.sort_values('importance', ascending=False)
                return None, importance_df
    
    # Try to find and load model
    models_dir = script_dir / "models"
    if models_dir.exists():
        model_files = [f for f in os.listdir(str(models_dir)) 
                       if f.startswith("rf_vortex_detector_") and f.endswith(".pkl")]
        
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = models_dir / latest_model
            print(f"  Loading model: {latest_model}")
            model = joblib.load(str(model_path))
            
            # Get feature names from training data
            train_file = script_dir / "datasets/train_features.csv"
            if train_file.exists():
                train_df = pd.read_csv(str(train_file), nrows=1)  # Just read header
                exclude_cols = ['window_id', 'label', 'event_sclk']
                feature_cols = [col for col in train_df.columns if col not in exclude_cols]
                
                # Verify features match expected ML dataset features
                if set(feature_cols) == set(EXPECTED_ML_FEATURES):
                    print(f"  [OK] Verified: Training features match ML dataset (15 features)")
                else:
                    print(f"  [WARNING] Training features may not match ML dataset")
                
                # Get importances from model
                importances = model.feature_importances_
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                print(f"  Computed feature importance from model")
                return model, importance_df
    
    raise FileNotFoundError(
        "Cannot find feature importance data. Please ensure results/feature_importance.csv exists.\n"
        "Expected 15 features for ML dataset model: " + ", ".join(EXPECTED_ML_FEATURES)
    )

def create_publication_feature_importance_plot(importance_df, output_path=None):
    """
    Create a publication-quality feature importance visualization.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Output file path (if None, uses script directory)
    """
    if output_path is None:
        script_dir = Path(__file__).parent.absolute()
        output_path = str(script_dir / "feature_importance_paper.png")
    
    print(f"\nCreating publication-quality feature importance plot...")
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=True)
    
    # Create figure with publication styling
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Create horizontal bar chart
    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color='#2E86AB', alpha=0.85, edgecolor='#1B4F72', linewidth=1.2)
    
    # Customize colors for top features
    n_features = len(importance_df)
    top_n = min(5, n_features)
    top_indices = importance_df.tail(top_n).index
    
    # Highlight top features with different color
    for idx in top_indices:
        bars[idx].set_color('#A23B72')
        bars[idx].set_alpha(0.9)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        # Format importance as percentage
        pct = importance * 100
        label_text = f'{pct:.1f}%'
        ax.text(importance + max(importance_df['importance']) * 0.02, 
                bar.get_y() + bar.get_height()/2, 
                label_text, 
                va='center', ha='left', 
                fontsize=LABEL_SIZE, 
                fontweight='bold',
                color='#1B4F72')
    
    # Customize axes
    ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Features', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Random Forest Feature Importance\n(ML Dataset Model - 15 Features)', 
                 fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    
    # Improve feature labels (replace underscores, capitalize)
    feature_labels = [feat.replace('_', ' ').title() for feat in importance_df['feature']]
    ax.set_yticklabels(feature_labels, fontsize=FONT_SIZE)
    
    # Grid styling
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, max(importance_df['importance']) * 1.25)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    # Add legend for top features
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A23B72', alpha=0.9, label='Top 5 Features'),
        Patch(facecolor='#2E86AB', alpha=0.85, label='Other Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=LABEL_SIZE-1, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved to: {output_path}")
    
    # Also save as PDF for publication
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved PDF to: {pdf_path}")
    
    plt.close()
    
    return output_path

def create_grouped_feature_importance(importance_df, output_path=None):
    """
    Create feature importance plot grouped by feature category.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        output_path: Output file path (if None, uses script directory)
    """
    if output_path is None:
        script_dir = Path(__file__).parent.absolute()
        output_path = str(script_dir / "feature_importance_grouped_paper.png")
    
    print(f"\nCreating grouped feature importance plot...")
    
    # Define feature categories
    feature_categories = {
        'Trend Features': ['overall_slope', 'first_half_slope', 'second_half_slope', 'trend_consistency'],
        'Pressure Drop Features': ['pressure_drop', 'drop_rate', 'min_position'],
        'Statistical Features': ['mean', 'std', 'range', 'first_half_mean', 'second_half_mean', 'mean_ratio'],
        'Anomaly Features': ['min_zscore', 'anomaly_strength']
    }
    
    # Assign categories
    importance_df['category'] = 'Other'
    for category, features in feature_categories.items():
        for feat in features:
            if feat in importance_df['feature'].values:
                importance_df.loc[importance_df['feature'] == feat, 'category'] = category
    
    # Sort by category and importance
    category_order = list(feature_categories.keys()) + ['Other']
    importance_df['category_order'] = importance_df['category'].map(
        {cat: i for i, cat in enumerate(category_order)}
    )
    importance_df = importance_df.sort_values(['category_order', 'importance'], ascending=[True, True])
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Color map for categories
    category_colors = {
        'Trend Features': '#E63946',
        'Pressure Drop Features': '#F77F00',
        'Statistical Features': '#2E86AB',
        'Anomaly Features': '#A23B72',
        'Other': '#6C757D'
    }
    
    # Create bars with category colors
    colors = [category_colors.get(cat, '#6C757D') for cat in importance_df['category']]
    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color=colors, alpha=0.85, edgecolor='white', linewidth=1.0)
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
        pct = importance * 100
        label_text = f'{pct:.1f}%'
        ax.text(importance + max(importance_df['importance']) * 0.02, 
                bar.get_y() + bar.get_height()/2, 
                label_text, 
                va='center', ha='left', 
                fontsize=LABEL_SIZE, 
                fontweight='bold')
    
    # Customize
    feature_labels = [feat.replace('_', ' ').title() for feat in importance_df['feature']]
    ax.set_yticklabels(feature_labels, fontsize=FONT_SIZE)
    ax.set_xlabel('Feature Importance', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Features', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Random Forest Feature Importance by Category\n(ML Dataset Model - 15 Features)', 
                 fontsize=TITLE_SIZE, fontweight='bold', pad=15)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_xlim(0, max(importance_df['importance']) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.85, label=cat) 
                      for cat, color in category_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=LABEL_SIZE-1, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"  Saved to: {output_path}")
    print(f"  Saved PDF to: {pdf_path}")
    
    plt.close()
    
    return output_path

def main():
    """Main execution function."""
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY FEATURE IMPORTANCE GRAPHS")
    print("=" * 70)
    
    try:
        # Load feature importance
        model, importance_df = load_model_and_features()
        
        print(f"\nFeature importance summary:")
        print(f"  Total features: {len(importance_df)}")
        print(f"  Model: Random Forest trained on ML dataset (ml_ready_vortex_data.csv)")
        print(f"  Feature set: 15 engineered features (no autoencoder features)")
        print(f"\n  Top 5 features:")
        top_5 = importance_df.head(5)
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"    {i}. {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")
        
        # Verify feature count
        if len(importance_df) != 15:
            print(f"\n  [WARNING] Expected 15 features for ML dataset model, found {len(importance_df)}")
        else:
            print(f"\n  [OK] Verified: Correct number of features (15) for ML dataset model")
        
        # Get script directory for output
        script_dir = Path(__file__).parent.absolute()
        
        # Create standard plot
        output1 = create_publication_feature_importance_plot(
            importance_df.copy(),
            str(script_dir / "feature_importance_paper.png")
        )
        
        # Create grouped plot
        output2 = create_grouped_feature_importance(
            importance_df.copy(),
            str(script_dir / "feature_importance_grouped_paper.png")
        )
        
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE GRAPHS GENERATED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  1. {output1} (and PDF)")
        print(f"  2. {output2} (and PDF)")
        print(f"\nBoth plots are publication-ready (300 DPI, PDF format available)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
