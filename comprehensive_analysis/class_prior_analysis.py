#!/usr/bin/env python3
"""
Class Prior Analysis for Mars Vortex Detection
===============================================
Analyzes class priors across train/val/test sets to inform class weight strategies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_class_distributions():
    """Load and analyze class distributions from all splits."""
    print("=" * 70)
    print("CLASS PRIOR ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    # 1. Training set (balanced - from train_features.csv)
    print("\n1. TRAINING SET (train_features.csv)")
    print("-" * 70)
    train_df = pd.read_csv("train_features.csv")
    train_labels = train_df['label'].values
    train_counts = np.bincount(train_labels)
    train_total = len(train_labels)
    train_prior_neg = train_counts[0] / train_total
    train_prior_pos = train_counts[1] / train_total
    
    results['train'] = {
        'counts': train_counts.tolist(),
        'total': train_total,
        'prior_negative': float(train_prior_neg),
        'prior_positive': float(train_prior_pos),
        'ratio_neg_to_pos': float(train_counts[0] / train_counts[1]) if train_counts[1] > 0 else np.inf,
        'source': 'train_features.csv (balanced)'
    }
    
    print(f"  Total samples: {train_total:,}")
    print(f"  Negative (0): {train_counts[0]:,} ({train_prior_neg*100:.2f}%)")
    print(f"  Positive (1): {train_counts[1]:,} ({train_prior_pos*100:.2f}%)")
    print(f"  Ratio: {train_counts[0]/train_counts[1]:.2f}:1 (Neg:Pos)")
    
    # 2. Validation set (natural imbalance - from val_sliding_features.csv)
    print("\n2. VALIDATION SET (val_sliding_features.csv)")
    print("-" * 70)
    val_df = pd.read_csv("val_sliding_features.csv")
    valid_val = val_df[val_df['label'] != 'Omit'].copy()
    valid_val['label'] = valid_val['label'].map({'True': 1, 'False': 0})
    val_labels = valid_val['label'].values
    val_counts = np.bincount(val_labels)
    val_total = len(val_labels)
    val_prior_neg = val_counts[0] / val_total
    val_prior_pos = val_counts[1] / val_total
    
    results['validation'] = {
        'counts': val_counts.tolist(),
        'total': val_total,
        'prior_negative': float(val_prior_neg),
        'prior_positive': float(val_prior_pos),
        'ratio_neg_to_pos': float(val_counts[0] / val_counts[1]) if val_counts[1] > 0 else np.inf,
        'source': 'val_sliding_features.csv (natural imbalance)'
    }
    
    print(f"  Total samples: {val_total:,}")
    print(f"  Negative (0): {val_counts[0]:,} ({val_prior_neg*100:.2f}%)")
    print(f"  Positive (1): {val_counts[1]:,} ({val_prior_pos*100:.2f}%)")
    print(f"  Ratio: {val_counts[0]/val_counts[1]:.2f}:1 (Neg:Pos)")
    
    # 3. Test set (natural imbalance - from test_sliding_features.csv)
    print("\n3. TEST SET (test_sliding_features.csv)")
    print("-" * 70)
    test_df = pd.read_csv("test_sliding_features.csv")
    valid_test = test_df[test_df['label'] != 'Omit'].copy()
    valid_test['label'] = valid_test['label'].map({'True': 1, 'False': 0})
    test_labels = valid_test['label'].values
    test_counts = np.bincount(test_labels)
    test_total = len(test_labels)
    test_prior_neg = test_counts[0] / test_total
    test_prior_pos = test_counts[1] / test_total
    
    results['test'] = {
        'counts': test_counts.tolist(),
        'total': test_total,
        'prior_negative': float(test_prior_neg),
        'prior_positive': float(test_prior_pos),
        'ratio_neg_to_pos': float(test_counts[0] / test_counts[1]) if test_counts[1] > 0 else np.inf,
        'source': 'test_sliding_features.csv (natural imbalance)'
    }
    
    print(f"  Total samples: {test_total:,}")
    print(f"  Negative (0): {test_counts[0]:,} ({test_prior_neg*100:.2f}%)")
    print(f"  Positive (1): {test_counts[1]:,} ({test_prior_pos*100:.2f}%)")
    print(f"  Ratio: {test_counts[0]/test_counts[1]:.2f}:1 (Neg:Pos)")
    
    return results

def calculate_class_weights(prior_neg, prior_pos, method='inverse_freq'):
    """
    Calculate class weights using different methods.
    
    Methods:
    - 'inverse_freq': Standard inverse frequency (n_samples / (n_classes * count))
    - 'balanced': sklearn's balanced (n_samples / (n_classes * count))
    - 'sqrt': Square root of inverse frequency
    - 'prior_adjusted': Adjust based on deployment prior
    """
    weights = {}
    
    if method == 'inverse_freq':
        # Standard inverse frequency
        weight_neg = 1.0 / (2 * prior_neg)
        weight_pos = 1.0 / (2 * prior_pos)
        # Normalize so they sum to 2 (for 2 classes)
        total = weight_neg + weight_pos
        weights[0] = weight_neg * 2 / total
        weights[1] = weight_pos * 2 / total
        
    elif method == 'balanced':
        # sklearn's balanced (same as inverse_freq for binary)
        weight_neg = 1.0 / (2 * prior_neg)
        weight_pos = 1.0 / (2 * prior_pos)
        total = weight_neg + weight_pos
        weights[0] = weight_neg * 2 / total
        weights[1] = weight_pos * 2 / total
        
    elif method == 'sqrt':
        # Square root of inverse frequency (less aggressive)
        weight_neg = np.sqrt(1.0 / (2 * prior_neg))
        weight_pos = np.sqrt(1.0 / (2 * prior_pos))
        total = weight_neg + weight_pos
        weights[0] = weight_neg * 2 / total
        weights[1] = weight_pos * 2 / total
        
    elif method == 'prior_adjusted':
        # Adjust weights to account for deployment prior
        # Weight positive class more if deployment has fewer positives
        deployment_ratio = prior_neg / prior_pos  # How many negatives per positive
        weight_neg = 1.0
        weight_pos = deployment_ratio  # Scale positive weight by imbalance ratio
        total = weight_neg + weight_pos
        weights[0] = weight_neg * 2 / total
        weights[1] = weight_pos * 2 / total
    
    return weights

def analyze_current_weights(train_prior_neg, train_prior_pos):
    """Analyze current class weight calculation."""
    print("\n" + "=" * 70)
    print("CURRENT CLASS WEIGHT CALCULATION")
    print("=" * 70)
    
    # Current method (from improved_train_rf_model.py)
    # weight_negative = total_samples / (2 * class_counts[0])
    # weight_positive = total_samples / (2 * class_counts[1])
    
    # For balanced data (1:1 ratio), this gives:
    # weight_negative = n / (2 * n/2) = n / n = 1.0
    # weight_positive = n / (2 * n/2) = n / n = 1.0
    
    # But we're training on balanced data, so let's show what happens:
    print("\nCurrent Method (from improved_train_rf_model.py):")
    print("  weight_negative = total_samples / (2 * class_counts[0])")
    print("  weight_positive = total_samples / (2 * class_counts[1])")
    print("\n  For balanced training data (1:1 ratio):")
    print("    Both weights ~= 1.0 (equal weighting)")
    print("\n  Problem: Model trained on balanced data but deployed on imbalanced data!")
    
    return None

def suggest_improved_weights(train_results, val_results, test_results):
    """Suggest improved class weights based on deployment priors."""
    print("\n" + "=" * 70)
    print("SUGGESTED CLASS WEIGHT STRATEGIES")
    print("=" * 70)
    
    # Use validation prior as deployment prior (closest to real-world)
    deployment_prior_neg = val_results['prior_negative']
    deployment_prior_pos = val_results['prior_positive']
    
    print(f"\nDeployment Prior (from validation set):")
    print(f"  Negative: {deployment_prior_neg*100:.2f}%")
    print(f"  Positive: {deployment_prior_pos*100:.2f}%")
    print(f"  Ratio: {deployment_prior_neg/deployment_prior_pos:.1f}:1")
    
    suggestions = {}
    
    # Strategy 1: Use deployment prior directly
    print("\n" + "-" * 70)
    print("Strategy 1: Weight by Deployment Prior")
    print("-" * 70)
    weights1 = calculate_class_weights(deployment_prior_neg, deployment_prior_pos, 'inverse_freq')
    suggestions['deployment_prior'] = weights1
    print(f"  Weight Negative (0): {weights1[0]:.4f}")
    print(f"  Weight Positive (1): {weights1[1]:.4f}")
    print(f"  Ratio: {weights1[1]/weights1[0]:.2f}:1 (Pos:Neg)")
    
    # Strategy 2: Square root (less aggressive)
    print("\n" + "-" * 70)
    print("Strategy 2: Square Root (Less Aggressive)")
    print("-" * 70)
    weights2 = calculate_class_weights(deployment_prior_neg, deployment_prior_pos, 'sqrt')
    suggestions['sqrt'] = weights2
    print(f"  Weight Negative (0): {weights2[0]:.4f}")
    print(f"  Weight Positive (1): {weights2[1]:.4f}")
    print(f"  Ratio: {weights2[1]/weights2[0]:.2f}:1 (Pos:Neg)")
    
    # Strategy 3: Prior-adjusted (moderate)
    print("\n" + "-" * 70)
    print("Strategy 3: Prior-Adjusted (Moderate)")
    print("-" * 70)
    weights3 = calculate_class_weights(deployment_prior_neg, deployment_prior_pos, 'prior_adjusted')
    suggestions['prior_adjusted'] = weights3
    print(f"  Weight Negative (0): {weights3[0]:.4f}")
    print(f"  Weight Positive (1): {weights3[1]:.4f}")
    print(f"  Ratio: {weights3[1]/weights3[0]:.2f}:1 (Pos:Neg)")
    
    return suggestions

def create_visualizations(results, suggestions):
    """Create visualization of class priors and weight strategies."""
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Class Prior Comparison (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    splits = ['Train\n(Balanced)', 'Validation\n(Natural)', 'Test\n(Natural)']
    neg_priors = [results['train']['prior_negative']*100, 
                  results['validation']['prior_negative']*100,
                  results['test']['prior_negative']*100]
    pos_priors = [results['train']['prior_positive']*100,
                  results['validation']['prior_positive']*100,
                  results['test']['prior_positive']*100]
    
    x = np.arange(len(splits))
    width = 0.35
    ax1.bar(x - width/2, neg_priors, width, label='Negative (0)', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, pos_priors, width, label='Positive (1)', color='#e74c3c', alpha=0.8)
    ax1.set_ylabel('Class Prior (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Class Prior Comparison Across Splits', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add percentage labels
    for i, (neg, pos) in enumerate(zip(neg_priors, pos_priors)):
        ax1.text(i - width/2, neg + 1, f'{neg:.1f}%', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, pos + 1, f'{pos:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Class Count Comparison (Log Scale)
    ax2 = fig.add_subplot(gs[0, 1])
    neg_counts = [results['train']['counts'][0], 
                  results['validation']['counts'][0],
                  results['test']['counts'][0]]
    pos_counts = [results['train']['counts'][1],
                  results['validation']['counts'][1],
                  results['test']['counts'][1]]
    
    x = np.arange(len(splits))
    ax2.bar(x - width/2, neg_counts, width, label='Negative (0)', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, pos_counts, width, label='Positive (1)', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Sample Count (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Class Count Comparison (Log Scale)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Imbalance Ratio Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ratios = [results['train']['ratio_neg_to_pos'],
              results['validation']['ratio_neg_to_pos'],
              results['test']['ratio_neg_to_pos']]
    
    bars = ax3.bar(splits, ratios, color=['#2ecc71', '#f39c12', '#9b59b6'], alpha=0.8)
    ax3.set_ylabel('Imbalance Ratio (Neg:Pos)', fontsize=12, fontweight='bold')
    ax3.set_title('Class Imbalance Ratio', fontsize=14, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add ratio labels
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}:1', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Suggested Class Weights
    ax4 = fig.add_subplot(gs[1, 1])
    strategies = list(suggestions.keys())
    neg_weights = [suggestions[s][0] for s in strategies]
    pos_weights = [suggestions[s][1] for s in strategies]
    
    x = np.arange(len(strategies))
    ax4.bar(x - width/2, neg_weights, width, label='Weight Negative (0)', color='#3498db', alpha=0.8)
    ax4.bar(x + width/2, pos_weights, width, label='Weight Positive (1)', color='#e74c3c', alpha=0.8)
    ax4.set_ylabel('Class Weight', fontsize=12, fontweight='bold')
    ax4.set_title('Suggested Class Weight Strategies', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.replace('_', ' ').title() for s in strategies], rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Weight Ratio Comparison
    ax5 = fig.add_subplot(gs[2, :])
    weight_ratios = [suggestions[s][1] / suggestions[s][0] for s in strategies]
    
    bars = ax5.bar(strategies, weight_ratios, color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.8)
    ax5.set_ylabel('Weight Ratio (Pos:Neg)', fontsize=12, fontweight='bold')
    ax5.set_title('Positive-to-Negative Weight Ratio by Strategy', fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)
    
    # Add ratio labels
    for bar, ratio in zip(bars, weight_ratios):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}:1', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('Class Prior Analysis for Mars Vortex Detection', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = f"results/class_prior_analysis_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    return output_path

def save_analysis_report(results, suggestions, output_path):
    """Save detailed analysis report."""
    report = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'class_priors': results,
        'suggested_weights': {k: {str(kk): float(vv) for kk, vv in v.items()} 
                              for k, v in suggestions.items()},
        'recommendations': {
            'current_issue': 'Model trained on balanced data (1:1) but deployed on imbalanced data (~150:1)',
            'recommended_strategy': 'deployment_prior',
            'rationale': 'Use validation set prior to calculate weights, as it represents real-world deployment conditions'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis report saved to: {output_path}")

def main():
    """Main execution function."""
    import os
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Load class distributions
    results = load_class_distributions()
    
    # Analyze current weights
    analyze_current_weights(
        results['train']['prior_negative'],
        results['train']['prior_positive']
    )
    
    # Suggest improved weights
    suggestions = suggest_improved_weights(
        results['train'], 
        results['validation'], 
        results['test']
    )
    
    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    viz_path = create_visualizations(results, suggestions)
    
    # Save report
    os.makedirs("results", exist_ok=True)
    report_path = f"results/class_prior_analysis_{timestamp}.json"
    save_analysis_report(results, suggestions, report_path)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Takeaways:")
    print(f"  1. Training set is balanced (1:1 ratio)")
    print(f"  2. Validation/Test sets are highly imbalanced ({results['validation']['ratio_neg_to_pos']:.0f}:1 ratio)")
    print(f"  3. Current weights don't account for deployment prior")
    print(f"  4. Recommended: Use deployment prior strategy")
    print(f"     class_weight={{0: {suggestions['deployment_prior'][0]:.4f}, 1: {suggestions['deployment_prior'][1]:.4f}}}")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

