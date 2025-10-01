#!/usr/bin/env python3
"""
Analysis script for LSTM Autoencoder output dataset.
Analyzes reconstruction errors, anomaly scores, and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class AutoencoderOutputAnalyzer:
    """Analyzer for autoencoder output dataset."""
    
    def __init__(self, window_size=60, debug=False):
        self.window_size = window_size
        self.debug = debug
        self.autoencoder = None
        self.results = None
        
    def debug_print(self, *args, **kwargs):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(*args, **kwargs)
    
    def load_autoencoder(self, model_path):
        """Load the trained autoencoder model."""
        print(f"Loading autoencoder from: {model_path}")
        self.autoencoder = tf.keras.models.load_model(model_path)
        print("Autoencoder loaded successfully")
        
        # Print model summary
        print("\nAutoencoder Model Summary:")
        self.autoencoder.summary()
    
    def load_data(self, data_path):
        """Load the dataset."""
        print(f"Loading data from: {data_path}")
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} samples")
        return data
    
    def prepare_sequences(self, data, apply_detrending=True):
        """Prepare sequences from data for analysis."""
        pressure_values = data['PRESSURE'].values
        gt_detection = data['gt_detection_win'].values
        gt_fwhm = data['gt_fwhm'].values
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        
        sequences_list = []
        labels_list = []
        indices_list = []
        
        self.debug_print(f"\nPreparing sequences for analysis...")
        
        # Process all data with sliding window
        for i in range(self.window_size, len(data)):
            # Get the pressure window
            pressure_window = pressure_values[i-self.window_size:i].copy()
            
            if apply_detrending:
                # Apply local detrending
                local_mean = np.mean(pressure_window)
                pressure_window = pressure_window - local_mean
            
            # Reshape for LSTM: (window_size, 1)
            sequence = pressure_window.reshape(-1, 1)
            sequences_list.append(sequence)
            
            # Determine label (1 if window contains vortex, 0 otherwise)
            label = 1 if np.any(gt_combined[i-self.window_size:i] == 1) else 0
            labels_list.append(label)
            indices_list.append(i)
        
        sequences = np.array(sequences_list)
        labels = np.array(labels_list)
        indices = np.array(indices_list)
        
        self.debug_print(f"Prepared {len(sequences)} sequences")
        self.debug_print(f"Vortex windows: {sum(labels)}")
        self.debug_print(f"Normal windows: {len(labels) - sum(labels)}")
        
        return sequences, labels, indices
    
    def analyze_reconstruction_errors(self, X, y, indices):
        """Analyze reconstruction errors and anomaly scores."""
        print("\n=== Reconstruction Error Analysis ===")
        
        # Get reconstruction errors
        reconstruction_errors = self.autoencoder.predict(X)
        mse_errors = np.mean((X - reconstruction_errors) ** 2, axis=(1, 2))
        
        # Calculate statistics
        print(f"Reconstruction Error Statistics:")
        print(f"  Mean: {np.mean(mse_errors):.6f}")
        print(f"  Std: {np.std(mse_errors):.6f}")
        print(f"  Min: {np.min(mse_errors):.6f}")
        print(f"  Max: {np.max(mse_errors):.6f}")
        print(f"  25th percentile: {np.percentile(mse_errors, 25):.6f}")
        print(f"  50th percentile: {np.percentile(mse_errors, 50):.6f}")
        print(f"  75th percentile: {np.percentile(mse_errors, 75):.6f}")
        print(f"  95th percentile: {np.percentile(mse_errors, 95):.6f}")
        print(f"  99th percentile: {np.percentile(mse_errors, 99):.6f}")
        
        # Analyze by class
        normal_errors = mse_errors[y == 0]
        vortex_errors = mse_errors[y == 1]
        
        print(f"\nBy Class:")
        print(f"  Normal windows ({len(normal_errors)}):")
        print(f"    Mean error: {np.mean(normal_errors):.6f}")
        print(f"    Std error: {np.std(normal_errors):.6f}")
        print(f"    Max error: {np.max(normal_errors):.6f}")
        
        print(f"  Vortex windows ({len(vortex_errors)}):")
        print(f"    Mean error: {np.mean(vortex_errors):.6f}")
        print(f"    Std error: {np.std(vortex_errors):.6f}")
        print(f"    Max error: {np.max(vortex_errors):.6f}")
        
        # Calculate separation metrics
        if len(vortex_errors) > 0:
            separation = (np.mean(vortex_errors) - np.mean(normal_errors)) / np.std(normal_errors)
            print(f"  Separation (vortex mean - normal mean) / normal std: {separation:.3f}")
        
        return mse_errors, reconstruction_errors
    
    def analyze_threshold_performance(self, mse_errors, y):
        """Analyze performance at different thresholds."""
        print("\n=== Threshold Performance Analysis ===")
        
        # Try different percentiles as thresholds
        percentiles = [50, 60, 70, 80, 85, 90, 95, 97, 99]
        results = []
        
        for percentile in percentiles:
            threshold = np.percentile(mse_errors, percentile)
            y_pred = (mse_errors > threshold).astype(int)
            
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            
            results.append({
                'percentile': percentile,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
            
            print(f"  {percentile}th percentile (threshold: {threshold:.6f}):")
            print(f"    Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"    TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        return pd.DataFrame(results)
    
    def plot_error_distributions(self, mse_errors, y):
        """Plot reconstruction error distributions."""
        print("\n=== Plotting Error Distributions ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall distribution
        axes[0, 0].hist(mse_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Reconstruction Error (MSE)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Overall Reconstruction Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log scale distribution
        axes[0, 1].hist(mse_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Reconstruction Error (MSE)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Reconstruction Error Distribution (Log Scale)')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # By class
        normal_errors = mse_errors[y == 0]
        vortex_errors = mse_errors[y == 1]
        
        axes[1, 0].hist(normal_errors, bins=30, alpha=0.7, label='Normal', edgecolor='black')
        axes[1, 0].hist(vortex_errors, bins=30, alpha=0.7, label='Vortex', edgecolor='black')
        axes[1, 0].set_xlabel('Reconstruction Error (MSE)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reconstruction Error by Class')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot
        data_for_box = [normal_errors, vortex_errors]
        labels = ['Normal', 'Vortex']
        axes[1, 1].boxplot(data_for_box, labels=labels)
        axes[1, 1].set_ylabel('Reconstruction Error (MSE)')
        axes[1, 1].set_title('Reconstruction Error Box Plot by Class')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('autoencoder_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Error distribution plots saved to: autoencoder_error_analysis.png")
    
    def plot_threshold_analysis(self, threshold_results):
        """Plot threshold performance analysis."""
        print("\n=== Plotting Threshold Analysis ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision, Recall, F1 vs threshold
        axes[0, 0].plot(threshold_results['percentile'], threshold_results['precision'], 
                        'o-', label='Precision', linewidth=2, markersize=8)
        axes[0, 0].plot(threshold_results['percentile'], threshold_results['recall'], 
                        's-', label='Recall', linewidth=2, markersize=8)
        axes[0, 0].plot(threshold_results['percentile'], threshold_results['f1'], 
                        '^-', label='F1-Score', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('Threshold Percentile')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Threshold values
        axes[0, 1].plot(threshold_results['percentile'], threshold_results['threshold'], 
                        'o-', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_xlabel('Threshold Percentile')
        axes[0, 1].set_ylabel('Threshold Value')
        axes[0, 1].set_title('Threshold Values vs Percentile')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix components
        axes[1, 0].plot(threshold_results['percentile'], threshold_results['tp'], 
                        'o-', label='True Positives', linewidth=2, markersize=8)
        axes[1, 0].plot(threshold_results['percentile'], threshold_results['fp'], 
                        's-', label='False Positives', linewidth=2, markersize=8)
        axes[1, 0].plot(threshold_results['percentile'], threshold_results['fn'], 
                        '^-', label='False Negatives', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('Threshold Percentile')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Confusion Matrix Components vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curve
        axes[1, 1].plot(threshold_results['recall'], threshold_results['precision'], 
                        'o-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add threshold labels
        for i, row in threshold_results.iterrows():
            axes[1, 1].annotate(f"{row['percentile']}%", 
                               (row['recall'], row['precision']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)
        
        plt.tight_layout()
        plt.savefig('autoencoder_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Threshold analysis plots saved to: autoencoder_threshold_analysis.png")
    
    def analyze_extreme_cases(self, X, y, mse_errors, indices):
        """Analyze extreme reconstruction error cases."""
        print("\n=== Extreme Cases Analysis ===")
        
        # Find extreme cases
        normal_indices = np.where(y == 0)[0]
        vortex_indices = np.where(y == 1)[0]
        
        # Highest error normal cases
        normal_errors = mse_errors[normal_indices]
        highest_normal_indices = normal_indices[np.argsort(normal_errors)[-10:]]
        
        # Lowest error vortex cases
        vortex_errors = mse_errors[vortex_indices]
        lowest_vortex_indices = vortex_indices[np.argsort(vortex_errors)[:10]]
        
        print(f"Top 10 highest error normal cases:")
        for i, idx in enumerate(highest_normal_indices):
            print(f"  {i+1}. Index {indices[idx]}, Error: {mse_errors[idx]:.6f}")
        
        print(f"\nTop 10 lowest error vortex cases:")
        for i, idx in enumerate(lowest_vortex_indices):
            print(f"  {i+1}. Index {indices[idx]}, Error: {mse_errors[idx]:.6f}")
        
        # Plot extreme cases
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        
        # Plot highest error normal cases
        for i, idx in enumerate(highest_normal_indices):
            original = X[idx, :, 0]
            reconstructed = self.autoencoder.predict(X[idx:idx+1])[0, :, 0]
            
            axes[0, i].plot(original, 'b-', label='Original', linewidth=2)
            axes[0, i].plot(reconstructed, 'r--', label='Reconstructed', linewidth=1)
            axes[0, i].set_title(f'Normal (Error: {mse_errors[idx]:.4f})')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # Plot lowest error vortex cases
        for i, idx in enumerate(lowest_vortex_indices):
            original = X[idx, :, 0]
            reconstructed = self.autoencoder.predict(X[idx:idx+1])[0, :, 0]
            
            axes[1, i].plot(original, 'b-', label='Original', linewidth=2)
            axes[1, i].plot(reconstructed, 'r--', label='Reconstructed', linewidth=1)
            axes[1, i].set_title(f'Vortex (Error: {mse_errors[idx]:.4f})')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('autoencoder_extreme_cases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Extreme cases plots saved to: autoencoder_extreme_cases.png")
    
    def generate_summary_report(self, mse_errors, y, threshold_results):
        """Generate a comprehensive summary report."""
        print("\n=== Summary Report ===")
        
        # Calculate overall metrics
        roc_auc = roc_auc_score(y, mse_errors)
        pr_auc = average_precision_score(y, mse_errors)
        
        # Find best threshold
        best_idx = threshold_results['f1'].idxmax()
        best_threshold = threshold_results.loc[best_idx]
        
        print(f"Overall Performance:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Best F1-Score: {best_threshold['f1']:.4f}")
        print(f"  Best Precision: {best_threshold['precision']:.4f}")
        print(f"  Best Recall: {best_threshold['recall']:.4f}")
        print(f"  Best Threshold: {best_threshold['threshold']:.6f} ({best_threshold['percentile']}th percentile)")
        
        # Class distribution
        vortex_prevalence = sum(y) / len(y)
        print(f"\nClass Distribution:")
        print(f"  Vortex windows: {sum(y)} ({vortex_prevalence:.3%})")
        print(f"  Normal windows: {len(y) - sum(y)} ({(1-vortex_prevalence):.3%})")
        
        # Error statistics by class
        normal_errors = mse_errors[y == 0]
        vortex_errors = mse_errors[y == 1]
        
        print(f"\nError Statistics by Class:")
        print(f"  Normal - Mean: {np.mean(normal_errors):.6f}, Std: {np.std(normal_errors):.6f}")
        print(f"  Vortex - Mean: {np.mean(vortex_errors):.6f}, Std: {np.std(vortex_errors):.6f}")
        
        if len(vortex_errors) > 0:
            separation = (np.mean(vortex_errors) - np.mean(normal_errors)) / np.std(normal_errors)
            print(f"  Separation: {separation:.3f} standard deviations")
        
        # Save detailed results
        results_df = pd.DataFrame({
            'index': range(len(mse_errors)),
            'reconstruction_error': mse_errors,
            'is_vortex': y,
            'predicted_anomaly': mse_errors > best_threshold['threshold']
        })
        
        results_df.to_csv('autoencoder_analysis_results.csv', index=False)
        print(f"\nDetailed results saved to: autoencoder_analysis_results.csv")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'best_threshold': best_threshold,
            'vortex_prevalence': vortex_prevalence,
            'normal_error_mean': np.mean(normal_errors),
            'normal_error_std': np.std(normal_errors),
            'vortex_error_mean': np.mean(vortex_errors),
            'vortex_error_std': np.std(vortex_errors)
        }

def main():
    """Main function to analyze autoencoder output."""
    
    parser = argparse.ArgumentParser(description='Analyze LSTM Autoencoder output dataset')
    parser.add_argument('--model_path', type=str, default='best_autoencoder.h5', 
                       help='Path to autoencoder model')
    parser.add_argument('--data_path', type=str, 
                       default='../../data/ml_ready_vortex_data.csv',
                       help='Path to data file')
    parser.add_argument('--window_size', type=int, default=60, help='Window size')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--data_reduction', type=float, default=1.0, 
                       help='Reduce dataset by this fraction (0.1 = 10% of data, 1.0 = full dataset)')
    
    args = parser.parse_args()
    
    print("=== LSTM Autoencoder Output Analysis ===")
    
    # Initialize analyzer
    analyzer = AutoencoderOutputAnalyzer(window_size=args.window_size, debug=args.debug)
    
    # Load autoencoder
    model_path = Path(__file__).parent / args.model_path
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    analyzer.load_autoencoder(model_path)
    
    # Load data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    data = analyzer.load_data(data_path)
    
    # Apply data reduction if specified
    if args.data_reduction < 1.0:
        reduction_size = int(len(data) * args.data_reduction)
        data = data.iloc[:reduction_size]
        print(f"Reduced dataset to {len(data)} samples ({args.data_reduction*100:.1f}% of original)")
    
    # Prepare sequences
    X, y, indices = analyzer.prepare_sequences(data, apply_detrending=True)
    
    # Analyze reconstruction errors
    mse_errors, reconstructions = analyzer.analyze_reconstruction_errors(X, y, indices)
    
    # Analyze threshold performance
    threshold_results = analyzer.analyze_threshold_performance(mse_errors, y)
    
    # Generate plots
    analyzer.plot_error_distributions(mse_errors, y)
    analyzer.plot_threshold_analysis(threshold_results)
    analyzer.analyze_extreme_cases(X, y, mse_errors, indices)
    
    # Generate summary report
    summary = analyzer.generate_summary_report(mse_errors, y, threshold_results)
    
    print(f"\nAnalysis complete! Generated files:")
    print(f"  - autoencoder_error_analysis.png")
    print(f"  - autoencoder_threshold_analysis.png")
    print(f"  - autoencoder_extreme_cases.png")
    print(f"  - autoencoder_analysis_results.csv")

if __name__ == "__main__":
    main() 