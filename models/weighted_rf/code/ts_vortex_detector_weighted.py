import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def create_temporal_features(sequence):
    """Create features from pressure difference sequence."""
    # Use earlier values for recent trend to avoid data leakage
    recent_trend = np.mean(sequence[-20:-10])  # Use values from 20-10 samples ago instead of last 10
    
    return np.array([
        np.mean(sequence),           # average pressure change
        np.std(sequence),            # variability
        np.sum(sequence < 0),        # number of pressure drops
        recent_trend,                # recent pressure trend (using earlier values)
        np.polyfit(range(len(sequence)), sequence, 1)[0]  # slope
    ])

def load_and_prepare_data(file_path, window_size=50):
    """Load and prepare data for training."""
    print("Loading data...", flush=True)
    start_time = time.time()
    df = pd.read_csv(file_path)
    df['pressure_diff'] = df['PRESSURE'].diff().shift(1)
    
    # Find vortex events (consecutive TRUE values in gt_fwhm)
    vortex_events = df[df['gt_fwhm'] > 0].index
    vortex_groups = []
    current_group = [vortex_events[0]]
    
    # Group consecutive vortex events
    for i in range(1, len(vortex_events)):
        if vortex_events[i] - vortex_events[i-1] == 1:
            current_group.append(vortex_events[i])
        else:
            vortex_groups.append(current_group)
            current_group = [vortex_events[i]]
    vortex_groups.append(current_group)
    
    # Prepare features and labels
    X = []
    y_detection = []
    y_fwhm = []
    
    # Process detection windows that precede vortex starts
    print("Processing detection windows that precede vortex starts...", flush=True)
    for vortex_group in tqdm(vortex_groups):
        vortex_start = vortex_group[0]
        
        # Find the detection window that precedes this vortex start
        preceding_detection = df[df['gt_detection_win'] > 0].index
        preceding_detection = preceding_detection[preceding_detection < vortex_start]
        
        if len(preceding_detection) > 0:
            # Use the last detection window before the vortex start
            detection_idx = preceding_detection[-1]
            
            if detection_idx >= window_size:
                sequence = df['pressure_diff'].iloc[detection_idx-window_size:detection_idx].values
                X.append(create_temporal_features(sequence))
                y_detection.append(1)
                y_fwhm.append(1)
    
    n_positive = len(y_detection)  # Number of detection windows
    n_negative = n_positive * 10   # Use 10x the number of detection windows
    
    # Process non-detection windows (using a reasonable subset)
    print("Processing non-detection windows...", flush=True)
    non_detection_mask = (df['gt_detection_win'] == 0) & (df.index >= window_size)
    valid_indices = df[non_detection_mask].index.values
    
    # Randomly select a subset of non-detection windows
    if len(valid_indices) > n_negative:
        selected_indices = np.random.choice(valid_indices, size=n_negative, replace=False)
        
        for idx in tqdm(selected_indices):
            sequence = df['pressure_diff'].iloc[idx-window_size:idx].values
            X.append(create_temporal_features(sequence))
            y_detection.append(0)
            y_fwhm.append(0)
    
    X = np.array(X)
    y_detection = np.array(y_detection)
    y_fwhm = np.array(y_fwhm)
    
    print(f"\nData preparation completed in {time.time() - start_time:.2f} seconds")
    print(f"Total sequences: {len(X)}")
    print(f"Detection windows preceding vortex starts: {np.sum(y_detection == 1)}")
    print(f"Non-detection windows: {np.sum(y_detection == 0)}")
    print(f"FWHM vortex events: {np.sum(y_fwhm == 1)}")
    print(f"Non-FWHM events: {np.sum(y_fwhm == 0)}")
    print(f"Class distribution: {np.mean(y_detection == 1)*100:.2f}% detection windows")
    
    return X, y_detection, y_fwhm

def analyze_pressure_patterns(df, y_true, y_pred, window_size=50):
    """Analyze pressure patterns in true positives, false positives, and false negatives."""
    patterns = {
        'true_positives': [],
        'false_positives': [],
        'false_negatives': []
    }
    
    # Get indices for each category
    true_positives = np.where((y_true == 1) & (y_pred == 1))[0]
    false_positives = np.where((y_true == 0) & (y_pred == 1))[0]
    false_negatives = np.where((y_true == 1) & (y_pred == 0))[0]
    
    # Analyze patterns for each category
    for category, indices in [('true_positives', true_positives), 
                            ('false_positives', false_positives),
                            ('false_negatives', false_negatives)]:
        if len(indices) > 0:  # Only process if we have examples
            for idx in indices:
                sequence = df['pressure_diff'].iloc[idx-window_size:idx].values
                if len(sequence) > 0:  # Only process if sequence is not empty
                    patterns[category].append({
                        'mean_change': np.mean(sequence),
                        'std_change': np.std(sequence),
                        'max_drop': np.min(sequence),
                        'n_drops': np.sum(sequence < 0),
                        'slope': np.polyfit(range(len(sequence)), sequence, 1)[0]
                    })
    
    # Calculate statistics for each category
    stats = {}
    for category, pattern_list in patterns.items():
        if pattern_list:  # if not empty
            stats[category] = {
                'mean_change': np.mean([p['mean_change'] for p in pattern_list]),
                'std_change': np.mean([p['std_change'] for p in pattern_list]),
                'max_drop': np.mean([p['max_drop'] for p in pattern_list]),
                'n_drops': np.mean([p['n_drops'] for p in pattern_list]),
                'slope': np.mean([p['slope'] for p in pattern_list])
            }
        else:
            stats[category] = {
                'mean_change': np.nan,
                'std_change': np.nan,
                'max_drop': np.nan,
                'n_drops': np.nan,
                'slope': np.nan
            }
    
    return stats

def visualize_model_metrics(model, X_test, y_test, y_pred, y_pred_proba, model_name, save_dir):
    """Create comprehensive visualizations for model metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # y_pred_proba is already 1D
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png')
    plt.close()
    
    # 3. Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)  # y_pred_proba is already 1D
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curve.png')
    plt.close()
    
    # 4. Feature Importance
    plt.figure(figsize=(10, 6))
    importance = model.feature_importances_
    feature_names = ['Mean Pressure Change', 'Pressure Variability', 'Number of Drops', 'Recent Trend', 'Overall Slope']
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45)
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png')
    plt.close()
    
    # 5. Prediction Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba, bins=50, alpha=0.75)  # y_pred_proba is already 1D
    plt.title(f'Prediction Probability Distribution - {model_name}')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_dir / 'probability_distribution.png')
    plt.close()
    
    # 6. Combined Metrics Plot
    metrics = {
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC': roc_auc
    }
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.title(f'Model Metrics - {model_name}')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_dir / 'combined_metrics.png')
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(save_dir / 'metrics_summary.csv', index=False)
    
    print(f"Visualizations saved to {save_dir}")

def create_model_report(model_name, results_dir):
    """Create a comprehensive HTML report for the model."""
    results_dir = Path(results_dir)
    
    # Read metrics
    metrics_df = pd.read_csv(results_dir / 'metrics_summary.csv')
    
    # Create HTML report
    html_content = f"""
    <html>
    <head>
        <title>{model_name} Model Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
            .visualization {{ margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{model_name} Model Report</h1>
            
            <h2>Model Metrics</h2>
            <div class="metrics">
                {''.join([f'<div class="metric-card"><h3>{col}</h3><p>{val:.4f}</p></div>' for col, val in metrics_df.iloc[0].items()])}
            </div>
            
            <h2>Visualizations</h2>
            <div class="visualization">
                <h3>Confusion Matrix</h3>
                <img src="confusion_matrix.png" alt="Confusion Matrix">
            </div>
            <div class="visualization">
                <h3>ROC Curve</h3>
                <img src="roc_curve.png" alt="ROC Curve">
            </div>
            <div class="visualization">
                <h3>Precision-Recall Curve</h3>
                <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
            </div>
            <div class="visualization">
                <h3>Feature Importance</h3>
                <img src="feature_importance.png" alt="Feature Importance">
            </div>
            <div class="visualization">
                <h3>Prediction Probability Distribution</h3>
                <img src="probability_distribution.png" alt="Probability Distribution">
            </div>
            <div class="visualization">
                <h3>Combined Metrics</h3>
                <img src="combined_metrics.png" alt="Combined Metrics">
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    with open(results_dir / 'model_report.html', 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {results_dir / 'model_report.html'}")

def main():
    print("Starting weighted vortex detection script...", flush=True)
    
    # Load and prepare data
    print("Loading and preparing data...", flush=True)
    data_path = 'data/ml_ready_vortex_data.csv'
    df = pd.read_csv(data_path)
    df['pressure_diff'] = df['PRESSURE'].diff().shift(1)  # Calculate pressure difference
    X, y_detection, y_fwhm = load_and_prepare_data(data_path)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_detection, test_size=0.2, random_state=42, stratify=y_detection)
    
    # Define different class weight scenarios
    weight_scenarios = [
        'balanced',  # sklearn's balanced weights
        {0: 1, 1: 2},  # 2x weight for detection class
        {0: 1, 1: 3},  # 3x weight for detection class
        {0: 1, 1: 4},  # 4x weight for detection class
        {0: 1, 1: 5},  # 5x weight for detection class
        {0: 1, 1: 10}  # 10x weight for detection class
    ]
    
    # Store results for comparison
    all_results = {}
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate for each weight scenario
    for weights in weight_scenarios:
        print(f"\nTesting with class weights: {weights}", flush=True)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=weights)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        energy_saved = (1 - np.mean(y_pred)) * 100
        data_quality = precision * 100
        
        # Store results
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'energy_saved': energy_saved,
            'data_quality': data_quality
        }
        all_results[str(weights)] = results
        
        # Create results directory
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize model metrics
        visualize_model_metrics(
            model=model,
            X_test=X_test_scaled,
            y_test=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            model_name=f'Weighted RF Model (weights: {weights})',
            save_dir=results_dir
        )
        
        # Create HTML report
        create_model_report(f'Weighted RF Model (weights: {weights})', results_dir)
        
        # Analyze and print pressure patterns
        pattern_stats = analyze_pressure_patterns(df, y_test, y_pred)
        print("\nPressure Pattern Analysis:")
        for category in ['true_positives', 'false_positives', 'false_negatives']:
            if category in pattern_stats:
                print(f"\n{category.replace('_', ' ').title()}:")
                stats = pattern_stats[category]
                print(f"Mean Pressure Change: {stats['mean_change']:.3f}")
                print(f"Pressure Variability: {stats['std_change']:.3f}")
                print(f"Max Pressure Drop: {stats['max_drop']:.3f}")
                print(f"Number of Drops: {stats['n_drops']:.1f}")
                print(f"Slope: {stats['slope']:.3f}")
        
        # Print results
        print(f"\nResults for weights {weights}:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"AUC: {auc_score:.3f}")
        print(f"Energy Saved: {energy_saved:.2f}%")
        print(f"Data Quality: {data_quality:.2f}%")
    
    # Compare results across different weights
    print("\nComparison of Results Across Different Weights:", flush=True)
    metrics = ['precision', 'recall', 'f1', 'energy_saved', 'data_quality', 'auc']
    for metric in metrics:
        print(f"\n{metric.capitalize()}:", flush=True)
        for weights, results in all_results.items():
            val = results[metric]
            print(f"{weights}: {val:.3f}", flush=True)
    
    print("\nScript completed successfully!", flush=True)

if __name__ == "__main__":
    main() 