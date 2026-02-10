#!/usr/bin/env python3
"""
Create IEEE Conference Paper Visualization for TABLE IV
========================================================
Converts TABLE IV into a professional single-graph visualization suitable for IEEE publication.
"""

import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

DPI = 300
FIG_SIZE = (10, 6)  # Single graph size

# Table IV data (from image description)
table_data = [
    # Selection, Label, Model, Adj. Prec., Adj. Recall, Threshold, PR AUC, Base Prec., x Imp.
    {'selection': 'Opt-Prec.', 'label': 'detection-only', 'model': 'w15_detection_80 recon',
     'adj_prec': 0.6667, 'adj_recall': 0.0426, 'threshold': 0.85, 'pr_auc': 0.056, 
     'base_prec': 0.00509, 'x_imp': 131.0},
    {'selection': 'Opt-F1', 'label': 'detection-only', 'model': 'w15_detection_80 unbalanced',
     'adj_prec': 0.3313, 'adj_recall': 0.5957, 'threshold': 0.30, 'pr_auc': 0.080, 
     'base_prec': 0.00509, 'x_imp': 65.1},
    {'selection': 'Opt-Prec.', 'label': 'both', 'model': 'w30_both_55 unbalanced',
     'adj_prec': 0.7500, 'adj_recall': 0.0465, 'threshold': 0.85, 'pr_auc': 0.280, 
     'base_prec': 0.00849, 'x_imp': 88.3},
    {'selection': 'Opt-F1', 'label': 'both', 'model': 'w15_both_52 unbalanced',
     'adj_prec': 0.2305, 'adj_recall': 0.4651, 'threshold': 0.40, 'pr_auc': 0.211, 
     'base_prec': 0.00849, 'x_imp': 27.2},
]

def create_table_iv_visualization():
    """Create publication-quality single-graph visualization of TABLE IV."""
    
    # Extract data
    opt_prec_det_prec = next((e['adj_prec'] for e in table_data if e['selection'] == 'Opt-Prec.' and e['label'] == 'detection-only'), 0)
    opt_prec_det_recall = next((e['adj_recall'] for e in table_data if e['selection'] == 'Opt-Prec.' and e['label'] == 'detection-only'), 0)
    opt_prec_det_imp = next((e['x_imp'] for e in table_data if e['selection'] == 'Opt-Prec.' and e['label'] == 'detection-only'), 0)
    
    opt_f1_det_prec = next((e['adj_prec'] for e in table_data if e['selection'] == 'Opt-F1' and e['label'] == 'detection-only'), 0)
    opt_f1_det_recall = next((e['adj_recall'] for e in table_data if e['selection'] == 'Opt-F1' and e['label'] == 'detection-only'), 0)
    opt_f1_det_imp = next((e['x_imp'] for e in table_data if e['selection'] == 'Opt-F1' and e['label'] == 'detection-only'), 0)
    
    opt_prec_both_prec = next((e['adj_prec'] for e in table_data if e['selection'] == 'Opt-Prec.' and e['label'] == 'both'), 0)
    opt_prec_both_recall = next((e['adj_recall'] for e in table_data if e['selection'] == 'Opt-Prec.' and e['label'] == 'both'), 0)
    opt_prec_both_imp = next((e['x_imp'] for e in table_data if e['selection'] == 'Opt-Prec.' and e['label'] == 'both'), 0)
    
    opt_f1_both_prec = next((e['adj_prec'] for e in table_data if e['selection'] == 'Opt-F1' and e['label'] == 'both'), 0)
    opt_f1_both_recall = next((e['adj_recall'] for e in table_data if e['selection'] == 'Opt-F1' and e['label'] == 'both'), 0)
    opt_f1_both_imp = next((e['x_imp'] for e in table_data if e['selection'] == 'Opt-F1' and e['label'] == 'both'), 0)
    
    # Create single figure with grouped bars
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    # Prepare data for grouped bar chart
    # X-axis: 4 configurations
    categories = ['Opt-Prec.\nDet-Only', 'Opt-F1\nDet-Only', 'Opt-Prec.\nBoth', 'Opt-F1\nBoth']
    x_pos = np.arange(len(categories))
    width = 0.25
    
    # Extract values for each metric
    precision_values = [opt_prec_det_prec, opt_f1_det_prec, opt_prec_both_prec, opt_f1_both_prec]
    recall_values = [opt_prec_det_recall, opt_f1_det_recall, opt_prec_both_recall, opt_f1_both_recall]
    improvement_values = [opt_prec_det_imp, opt_f1_det_imp, opt_prec_both_imp, opt_f1_both_imp]
    
    # Normalize improvement values to fit on same scale (divide by max to get 0-1 range)
    max_imp = max(improvement_values)
    improvement_normalized = [v / max_imp for v in improvement_values]
    
    # Create grouped bars
    bars1 = ax.bar(x_pos - width, precision_values, width, label='Adjusted Precision', 
                   color='#2E86AB', alpha=0.85, edgecolor='#1B4F72', linewidth=1.2)
    bars2 = ax.bar(x_pos, recall_values, width, label='Adjusted Recall', 
                   color='#A23B72', alpha=0.85, edgecolor='#7A1F5C', linewidth=1.2)
    bars3 = ax.bar(x_pos + width, improvement_normalized, width, label='Improvement Factor (normalized)', 
                   color='#F18F01', alpha=0.85, edgecolor='#C66E00', linewidth=1.2)
    
    # Add value labels on precision bars
    for i, (bar, val) in enumerate(zip(bars1, precision_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold', color='#1B4F72')
    
    # Add value labels on recall bars
    for i, (bar, val) in enumerate(zip(bars2, recall_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold', color='#7A1F5C')
    
    # Add value labels on improvement bars (show actual x Imp values)
    for i, (bar, val, imp_val) in enumerate(zip(bars3, improvement_normalized, improvement_values)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                   f'{imp_val:.1f}x', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold', color='#C66E00')
    
    # Customize axes
    ax.set_ylabel('Score / Normalized Value', fontweight='bold', fontsize=11)
    ax.set_xlabel('Model Configuration', fontweight='bold', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    
    # Add secondary y-axis for improvement factor (actual values)
    ax2 = ax.twinx()
    ax2.set_ylabel('Improvement Factor (×)', fontweight='bold', fontsize=11, color='#C66E00')
    ax2.set_ylim([0, max_imp * 1.15])
    ax2.tick_params(axis='y', labelcolor='#C66E00', labelsize=9)
    
    # Add annotation explaining normalization
    ax.text(0.02, 0.98, 'Note: Improvement Factor (×) shown on right axis', 
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main Title
    fig.suptitle('TABLE IV: Optimal-Threshold Winners Under Detection-Only Evaluation\n' +
                 '(GT Detection Win)', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    output_path = 'table_iv_ieee_visualization.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n[SUCCESS] Visualization saved to: {output_path}")
    print(f"  Resolution: {FIG_SIZE[0]*DPI} x {FIG_SIZE[1]*DPI} pixels")
    print(f"  Format: PNG (IEEE publication quality)")
    
    # Also save as PDF for vector graphics
    output_path_pdf = 'table_iv_ieee_visualization.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"[SUCCESS] PDF version saved to: {output_path_pdf}")
    
    plt.close()
    
    return output_path, output_path_pdf

if __name__ == "__main__":
    print("=" * 70)
    print("Creating IEEE Conference Paper Visualization for TABLE IV")
    print("=" * 70)
    
    png_path, pdf_path = create_table_iv_visualization()
    
    print("\n" + "=" * 70)
    print("SUCCESS: Visualization created!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {png_path} (PNG, 300 DPI)")
    print(f"  2. {pdf_path} (PDF, vector graphics)")
    print(f"\nNote: These results are from a different model/system")
    print(f"      and should not be confused with the RF model results.")
