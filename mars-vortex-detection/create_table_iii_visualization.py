#!/usr/bin/env python3
"""
Create IEEE Conference Paper Visualization for TABLE III
=========================================================
Converts TABLE III into a professional graph suitable for IEEE publication.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.titlesize'] = 12

DPI = 300
FIG_SIZE = (9, 6)  # Reduced from (12, 8) for more compact visualization

# Table III data (from image description)
table_data = [
    # Event recall, Label, Model, Adj. Prec., Adj. Recall, Threshold, PR AUC, Base Prec., x Imp.
    {'event_recall': '≥ 0.10', 'label': 'both', 'model': 'w30_both_55 unbalanced recon',
     'adj_prec': 0.588, 'adj_recall': 0.116, 'threshold': 0.75, 'pr_auc': 0.280, 
     'base_prec': 0.00849, 'x_imp': 69.3},
    {'event_recall': '≥ 0.10', 'label': 'detection-only', 'model': 'w30_detection_50 unbalanced recon',
     'adj_prec': 0.341, 'adj_recall': 0.279, 'threshold': 0.35, 'pr_auc': 0.049, 
     'base_prec': 0.00509, 'x_imp': 67.1},
    {'event_recall': '≥ 0.20', 'label': 'both', 'model': 'w15_both_52 unbalanced recon',
     'adj_prec': 0.291, 'adj_recall': 0.209, 'threshold': 0.55, 'pr_auc': 0.211, 
     'base_prec': 0.00849, 'x_imp': 34.3},
    {'event_recall': '≥ 0.20', 'label': 'detection-only', 'model': 'w30_detection_50 unbalanced recon',
     'adj_prec': 0.341, 'adj_recall': 0.279, 'threshold': 0.35, 'pr_auc': 0.049, 
     'base_prec': 0.00509, 'x_imp': 67.1},
    {'event_recall': '≥ 0.30', 'label': 'both', 'model': 'w15_both_52 unbalanced',
     'adj_prec': 0.291, 'adj_recall': 0.302, 'threshold': 0.50, 'pr_auc': 0.211, 
     'base_prec': 0.00849, 'x_imp': 34.2},
    {'event_recall': '≥ 0.30', 'label': 'detection-only', 'model': 'w15_detection_80 unbalanced',
     'adj_prec': 0.331, 'adj_recall': 0.596, 'threshold': 0.30, 'pr_auc': 0.080, 
     'base_prec': 0.00509, 'x_imp': 65.1},
    {'event_recall': '≥ 0.40', 'label': 'both', 'model': 'w15_both_80 unbalanced',
     'adj_prec': 0.243, 'adj_recall': 0.419, 'threshold': 0.40, 'pr_auc': 0.335, 
     'base_prec': 0.00849, 'x_imp': 28.6},
    {'event_recall': '≥ 0.40', 'label': 'detection-only', 'model': 'w15_detection_80 unbalanced',
     'adj_prec': 0.331, 'adj_recall': 0.596, 'threshold': 0.30, 'pr_auc': 0.080, 
     'base_prec': 0.00509, 'x_imp': 65.1},
]

def create_table_iii_visualization():
    """Create publication-quality visualization of TABLE III."""
    
    # Organize data by event recall threshold
    event_recalls = ['≥ 0.10', '≥ 0.20', '≥ 0.30', '≥ 0.40']
    
    # Extract data for each threshold
    both_prec = []
    both_recall = []
    both_imp = []
    det_prec = []
    det_recall = []
    det_imp = []
    
    for er in event_recalls:
        # Find entries for this event recall
        both_entry = next((e for e in table_data if e['event_recall'] == er and e['label'] == 'both'), None)
        det_entry = next((e for e in table_data if e['event_recall'] == er and e['label'] == 'detection-only'), None)
        
        if both_entry:
            both_prec.append(both_entry['adj_prec'])
            both_recall.append(both_entry['adj_recall'])
            both_imp.append(both_entry['x_imp'])
        else:
            both_prec.append(0)
            both_recall.append(0)
            both_imp.append(0)
            
        if det_entry:
            det_prec.append(det_entry['adj_prec'])
            det_recall.append(det_entry['adj_recall'])
            det_imp.append(det_entry['x_imp'])
        else:
            det_prec.append(0)
            det_recall.append(0)
            det_imp.append(0)
    
    # Create figure with subplots
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25, 
                         left=0.10, right=0.95, top=0.90, bottom=0.10)
    
    x = np.arange(len(event_recalls))
    width = 0.35
    
    # Color scheme for IEEE papers (professional, accessible)
    color_both = '#2E86AB'  # Blue
    color_det = '#A23B72'   # Purple/Magenta
    color_both_light = '#6BAED6'
    color_det_light = '#C77C9F'
    
    # ============================================================================
    # SUBPLOT 1: Adjusted Precision
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    bars1a = ax1.bar(x - width/2, both_prec, width, label='Both Labels', 
                     color=color_both, alpha=0.85, edgecolor='#1B4F72', linewidth=1.2)
    bars1b = ax1.bar(x + width/2, det_prec, width, label='Detection-Only', 
                     color=color_det, alpha=0.85, edgecolor='#7A1F5C', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1a, bars1b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=7, fontweight='bold')
    
    ax1.set_ylabel('Adjusted Precision', fontweight='bold', fontsize=10)
    ax1.set_xlabel('Minimum Event-Recall Threshold', fontweight='bold', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(event_recalls)
    ax1.set_ylim([0, max(max(both_prec), max(det_prec)) * 1.15])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_title('(a) Adjusted Precision', fontweight='bold', fontsize=11, pad=8)
    
    # ============================================================================
    # SUBPLOT 2: Adjusted Recall
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    bars2a = ax2.bar(x - width/2, both_recall, width, label='Both Labels', 
                     color=color_both, alpha=0.85, edgecolor='#1B4F72', linewidth=1.2)
    bars2b = ax2.bar(x + width/2, det_recall, width, label='Detection-Only', 
                     color=color_det, alpha=0.85, edgecolor='#7A1F5C', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', 
                        fontsize=7, fontweight='bold')
    
    ax2.set_ylabel('Adjusted Recall', fontweight='bold', fontsize=10)
    ax2.set_xlabel('Minimum Event-Recall Threshold', fontweight='bold', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(event_recalls)
    ax2.set_ylim([0, max(max(both_recall), max(det_recall)) * 1.15])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_title('(b) Adjusted Recall', fontweight='bold', fontsize=11, pad=8)
    
    # ============================================================================
    # SUBPLOT 3: Improvement Factor (x Imp.)
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    bars3a = ax3.bar(x - width/2, both_imp, width, label='Both Labels', 
                     color=color_both, alpha=0.85, edgecolor='#1B4F72', linewidth=1.2)
    bars3b = ax3.bar(x + width/2, det_imp, width, label='Detection-Only', 
                     color=color_det, alpha=0.85, edgecolor='#7A1F5C', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars3a, bars3b]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{height:.1f}x', ha='center', va='bottom', 
                        fontsize=7, fontweight='bold')
    
    ax3.set_ylabel('Improvement Factor (×)', fontweight='bold', fontsize=10)
    ax3.set_xlabel('Minimum Event-Recall Threshold', fontweight='bold', fontsize=10)
    ax3.set_xticks(x)
    ax3.set_xticklabels(event_recalls)
    ax3.set_ylim([0, max(max(both_imp), max(det_imp)) * 1.15])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.set_title('(c) Improvement Over Baseline', fontweight='bold', fontsize=11, pad=8)
    
    # ============================================================================
    # SUBPLOT 4: Precision-Recall Trade-off (Scatter)
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Plot both labels
    both_mask = np.array(both_prec) > 0
    ax4.scatter([both_recall[i] for i in range(len(both_recall)) if both_mask[i]],
                [both_prec[i] for i in range(len(both_prec)) if both_mask[i]],
                s=150, color=color_both, marker='o', alpha=0.7, 
                edgecolors='#1B4F72', linewidths=2, label='Both Labels', zorder=3)
    
    # Plot detection-only
    det_mask = np.array(det_prec) > 0
    ax4.scatter([det_recall[i] for i in range(len(det_recall)) if det_mask[i]],
                [det_prec[i] for i in range(len(det_prec)) if det_mask[i]],
                s=150, color=color_det, marker='s', alpha=0.7, 
                edgecolors='#7A1F5C', linewidths=2, label='Detection-Only', zorder=3)
    
    # Add threshold labels
    for i, er in enumerate(event_recalls):
        if both_mask[i]:
            ax4.annotate(er, (both_recall[i], both_prec[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=7, color='#1B4F72', fontweight='bold')
        if det_mask[i]:
            ax4.annotate(er, (det_recall[i], det_prec[i]), 
                        xytext=(5, -10), textcoords='offset points', 
                        fontsize=7, color='#7A1F5C', fontweight='bold')
    
    ax4.set_xlabel('Adjusted Recall', fontweight='bold', fontsize=10)
    ax4.set_ylabel('Adjusted Precision', fontweight='bold', fontsize=10)
    ax4.set_xlim([0, max(max(both_recall), max(det_recall)) * 1.1])
    ax4.set_ylim([0, max(max(both_prec), max(det_prec)) * 1.1])
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.legend(loc='lower left', framealpha=0.9)
    ax4.set_title('(d) Precision-Recall Trade-off', fontweight='bold', fontsize=11, pad=8)
    
    # ============================================================================
    # Main Title
    # ============================================================================
    fig.suptitle('TABLE III: Best Models Under Minimum Event-Recall Thresholds\n' +
                 'Detection-Only Evaluation (GT Detection Win)', 
                 fontsize=12, fontweight='bold', y=0.97)
    
    # Save figure
    output_path = 'table_iii_ieee_visualization.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\n[SUCCESS] Visualization saved to: {output_path}")
    print(f"  Resolution: {FIG_SIZE[0]*DPI} x {FIG_SIZE[1]*DPI} pixels")
    print(f"  Format: PNG (IEEE publication quality)")
    
    # Also save as PDF for vector graphics
    output_path_pdf = 'table_iii_ieee_visualization.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"[SUCCESS] PDF version saved to: {output_path_pdf}")
    
    plt.close()
    
    return output_path, output_path_pdf

if __name__ == "__main__":
    print("=" * 70)
    print("Creating IEEE Conference Paper Visualization for TABLE III")
    print("=" * 70)
    
    png_path, pdf_path = create_table_iii_visualization()
    
    print("\n" + "=" * 70)
    print("SUCCESS: Visualization created!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {png_path} (PNG, 300 DPI)")
    print(f"  2. {pdf_path} (PDF, vector graphics)")
    print(f"\nNote: These results are from a different model/system")
    print(f"      and should not be confused with the RF model results.")
