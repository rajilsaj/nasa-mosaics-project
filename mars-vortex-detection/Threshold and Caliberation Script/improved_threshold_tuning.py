#!/usr/bin/env python3
"""
Improved Threshold Tuning for Extreme Class Imbalance
Finds optimal threshold for 99.4% negative class distribution
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

def load_model_and_validation_data():
    """Load trained model and validation data."""
    print("=" * 70)
    print("IMPROVED THRESHOLD TUNING FOR EXTREME IMBALANCE")
    print("=" * 70)
    
    # Find latest model
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.startswith("improved_rf_vortex_detector_") and f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("No improved model found. Run improved_train_rf_model.py first.")
    
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(models_dir, latest_model)
    print(f"Loading model: {latest_model}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Load validation features
    val_features_df = pd.read_csv("val_sliding_features.csv")
    print(f"Loaded {len(val_features_df)} validation feature vectors")
    
    # Prepare validation data
    valid_df = val_features_df[val_features_df['label'] != 'Omit'].copy()
    valid_df['label'] = valid_df['label'].map({'True': 1, 'False': 0})
    
    # Get feature columns, excluding metadata columns
    feature_cols = [col for col in valid_df.columns if col not in ['window_id', 'start_idx', 'end_idx', 'start_sclk', 'end_sclk', 'label']]
    
    # Load training features to get the exact feature order
    train_df = pd.read_csv("train_features.csv")
    train_feature_cols = [col for col in train_df.columns if col not in ['window_id', 'label']]
    
    # Ensure validation features match training features (remove any extra features)
    feature_cols = [col for col in train_feature_cols if col in feature_cols]
    
    print(f"Training features: {len(train_feature_cols)} - {train_feature_cols}")
    print(f"Validation features: {len(feature_cols)} - {feature_cols}")
    
    X_val = valid_df[feature_cols].values
    y_val = valid_df['label'].values
    
    print(f"Validation samples: {len(X_val)}")
    print(f"Class distribution: {np.bincount(y_val)}")
    print(f"Class ratio: {np.bincount(y_val)[0] / np.bincount(y_val)[1]:.1f}:1 (Negative:Positive)")
    
    return model, X_val, y_val, feature_cols, valid_df

def find_optimal_thresholds(y_true, y_proba):
    """Find optimal thresholds for different mission objectives."""
    print("\n" + "=" * 70)
    print("THRESHOLD OPTIMIZATION ANALYSIS")
    print("=" * 70)
    
    # Calculate precision-recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
    
    # Test different thresholds
    test_thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    for threshold in test_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision_val = precision_score(y_true, y_pred, zero_division=0)
        recall_val = recall_score(y_true, y_pred, zero_division=0)
        f1_val = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate false positive rate
        tn, fp, fn, tp = np.bincount(y_true * 2 + y_pred, minlength=4)
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_val,
            'fpr': fpr_val
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds for different objectives
    optimal_thresholds = {}
    
    # 1. Maximum F1-Score
    max_f1_idx = results_df['f1_score'].idxmax()
    optimal_thresholds['max_f1'] = {
        'threshold': results_df.loc[max_f1_idx, 'threshold'],
        'precision': results_df.loc[max_f1_idx, 'precision'],
        'recall': results_df.loc[max_f1_idx, 'recall'],
        'f1_score': results_df.loc[max_f1_idx, 'f1_score'],
        'fpr': results_df.loc[max_f1_idx, 'fpr']
    }
    
    # 2. High Precision (>= 0.1)
    high_precision_df = results_df[results_df['precision'] >= 0.1]
    if not high_precision_df.empty:
        best_precision_idx = high_precision_df['recall'].idxmax()
        optimal_thresholds['high_precision'] = {
            'threshold': results_df.loc[best_precision_idx, 'threshold'],
            'precision': results_df.loc[best_precision_idx, 'precision'],
            'recall': results_df.loc[best_precision_idx, 'recall'],
            'f1_score': results_df.loc[best_precision_idx, 'f1_score'],
            'fpr': results_df.loc[best_precision_idx, 'fpr']
        }
    
    # 3. High Recall (>= 0.5)
    high_recall_df = results_df[results_df['recall'] >= 0.5]
    if not high_recall_df.empty:
        best_recall_idx = high_recall_df['precision'].idxmax()
        optimal_thresholds['high_recall'] = {
            'threshold': results_df.loc[best_recall_idx, 'threshold'],
            'precision': results_df.loc[best_recall_idx, 'precision'],
            'recall': results_df.loc[best_recall_idx, 'recall'],
            'f1_score': results_df.loc[best_recall_idx, 'f1_score'],
            'fpr': results_df.loc[best_recall_idx, 'fpr']
        }
    
    # 4. Balanced Precision-Recall
    balanced_df = results_df.copy()
    balanced_df['balance_score'] = 1 - abs(balanced_df['precision'] - balanced_df['recall'])
    best_balance_idx = balanced_df['balance_score'].idxmax()
    optimal_thresholds['balanced'] = {
        'threshold': results_df.loc[best_balance_idx, 'threshold'],
        'precision': results_df.loc[best_balance_idx, 'precision'],
        'recall': results_df.loc[best_balance_idx, 'recall'],
        'f1_score': results_df.loc[best_balance_idx, 'f1_score'],
        'fpr': results_df.loc[best_balance_idx, 'fpr']
    }
    
    return optimal_thresholds, results_df, precision, recall, pr_thresholds

def plot_threshold_analysis(results_df, precision, recall, pr_thresholds, output_dir="results"):
    """Plot threshold analysis results."""
    print(f"\nCreating threshold analysis plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Threshold Optimization Analysis for Extreme Class Imbalance', fontsize=16, fontweight='bold')
    
    # 1. Precision-Recall Curve
    axes[0, 0].plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. F1-Score vs Threshold
    axes[0, 1].plot(results_df['threshold'], results_df['f1_score'], 'g-', linewidth=2, label='F1-Score')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_title('F1-Score vs Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Precision vs Threshold
    axes[1, 0].plot(results_df['threshold'], results_df['precision'], 'r-', linewidth=2, label='Precision')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Recall vs Threshold
    axes[1, 1].plot(results_df['threshold'], results_df['recall'], 'm-', linewidth=2, label='Recall')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(output_dir, f"threshold_optimization_{timestamp}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Threshold analysis plot saved to: {plot_filename}")
    return plot_filename

def print_threshold_recommendations(optimal_thresholds):
    """Print threshold recommendations for different mission objectives."""
    print("\n" + "=" * 70)
    print("THRESHOLD RECOMMENDATIONS FOR MISSION OBJECTIVES")
    print("=" * 70)
    
    print("\n1. MAXIMUM F1-SCORE (Balanced Performance):")
    if 'max_f1' in optimal_thresholds:
        t = optimal_thresholds['max_f1']
        print(f"   Threshold: {t['threshold']:.3f}")
        print(f"   Precision: {t['precision']:.3f}")
        print(f"   Recall:    {t['recall']:.3f}")
        print(f"   F1-Score:  {t['f1_score']:.3f}")
        print(f"   Use Case:  General purpose, balanced precision/recall")
    
    print("\n2. HIGH PRECISION (Minimize False Positives):")
    if 'high_precision' in optimal_thresholds:
        t = optimal_thresholds['high_precision']
        print(f"   Threshold: {t['threshold']:.3f}")
        print(f"   Precision: {t['precision']:.3f}")
        print(f"   Recall:    {t['recall']:.3f}")
        print(f"   F1-Score:  {t['f1_score']:.3f}")
        print(f"   Use Case:  Energy-limited operations, minimize false alarms")
    else:
        print("   No threshold found with precision >= 0.1")
    
    print("\n3. HIGH RECALL (Catch All Vortices):")
    if 'high_recall' in optimal_thresholds:
        t = optimal_thresholds['high_recall']
        print(f"   Threshold: {t['threshold']:.3f}")
        print(f"   Precision: {t['precision']:.3f}")
        print(f"   Recall:    {t['recall']:.3f}")
        print(f"   F1-Score:  {t['f1_score']:.3f}")
        print(f"   Use Case:  Science priority, don't miss any vortices")
    else:
        print("   No threshold found with recall >= 0.5")
    
    print("\n4. BALANCED PRECISION-RECALL:")
    if 'balanced' in optimal_thresholds:
        t = optimal_thresholds['balanced']
        print(f"   Threshold: {t['threshold']:.3f}")
        print(f"   Precision: {t['precision']:.3f}")
        print(f"   Recall:    {t['recall']:.3f}")
        print(f"   F1-Score:  {t['f1_score']:.3f}")
        print(f"   Use Case:  Balanced approach, equal weight to precision/recall")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEPS:")
    print("=" * 70)
    print("1. Choose threshold based on mission objectives")
    print("2. Run test evaluation with chosen threshold")
    print("3. Implement temporal logic for deployment")
    print("4. Consider ensemble methods for further improvement")
    print("=" * 70)

def save_threshold_results(optimal_thresholds, results_df, output_dir="results"):
    """Save threshold tuning results."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_filename = os.path.join(output_dir, f"threshold_results_{timestamp}.csv")
    results_df.to_csv(results_filename, index=False)
    print(f"Detailed threshold results saved to: {results_filename}")
    
    # Save optimal thresholds
    thresholds_filename = os.path.join(output_dir, f"optimal_thresholds_{timestamp}.json")
    with open(thresholds_filename, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"Optimal thresholds saved to: {thresholds_filename}")
    
    return results_filename, thresholds_filename

def main():
    """Main execution function."""
    print("Starting improved threshold tuning for extreme class imbalance...")
    
    # Step 1: Load model and validation data
    model, X_val, y_val, feature_cols, valid_df = load_model_and_validation_data()
    
    # Step 2: Get model probabilities
    print("\nGenerating model probabilities...")
    y_proba = model.predict_proba(X_val)[:, 1]
    print(f"Probability range: {y_proba.min():.4f} - {y_proba.max():.4f}")
    
    # Step 3: Find optimal thresholds
    optimal_thresholds, results_df, precision, recall, pr_thresholds = find_optimal_thresholds(y_val, y_proba)
    
    # Step 4: Plot analysis
    plot_filename = plot_threshold_analysis(results_df, precision, recall, pr_thresholds)
    
    # Step 5: Print recommendations
    print_threshold_recommendations(optimal_thresholds)
    
    # Step 6: Save results
    results_filename, thresholds_filename = save_threshold_results(optimal_thresholds, results_df)
    
    print(f"\nThreshold tuning completed successfully!")
    print(f"Next step: Run test evaluation with chosen threshold")
    print(f"Command: python improved_test_evaluation.py --threshold <chosen_threshold>")

if __name__ == "__main__":
    main()
