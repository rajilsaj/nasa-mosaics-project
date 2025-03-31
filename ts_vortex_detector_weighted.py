import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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
    print("Loading data...")
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
    print("Processing detection windows that precede vortex starts...")
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
    print("Processing non-detection windows...")
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

def perform_cross_validation(X, y_detection, y_fwhm, class_weights, df, n_splits=5):
    """Perform stratified k-fold cross-validation with class weights."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'energy_saved': [],
        'data_quality': [],
        'pattern_stats': []
    }
    
    feature_importances = []
    
    print(f"\nPerforming {n_splits}-fold cross-validation with class weights {class_weights}...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_detection), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_detection, y_test_detection = y_detection[train_idx], y_detection[test_idx]
        y_train_fwhm, y_test_fwhm = y_fwhm[train_idx], y_fwhm[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weights)
        model.fit(X_train_scaled, y_train_detection)
        
        feature_importances.append(model.feature_importances_)
        
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        precision = precision_score(y_test_detection, y_pred)
        recall = recall_score(y_test_detection, y_pred)
        f1 = f1_score(y_test_detection, y_pred)
        
        energy_saved = (1 - np.mean(y_pred)) * 100
        data_quality = precision * 100
        
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        cv_results['f1'].append(f1)
        cv_results['energy_saved'].append(energy_saved)
        cv_results['data_quality'].append(data_quality)
        
        # Analyze pressure patterns for this fold
        pattern_stats = analyze_pressure_patterns(df, y_test_detection, y_pred)
        cv_results['pattern_stats'].append(pattern_stats)
        
        print(f"Fold {fold} Results:")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-score: {f1:.3f}")
        print(f"Energy Saved: {energy_saved:.2f}%")
        print(f"Data Quality: {data_quality:.2f}%")
    
    print("\nCross-validation Results:")
    print(f"Average Precision: {np.mean(cv_results['precision']):.3f} (±{np.std(cv_results['precision']):.3f})")
    print(f"Average Recall: {np.mean(cv_results['recall']):.3f} (±{np.std(cv_results['recall']):.3f})")
    print(f"Average F1-score: {np.mean(cv_results['f1']):.3f} (±{np.std(cv_results['f1']):.3f})")
    print(f"Average Energy Saved: {np.mean(cv_results['energy_saved']):.2f}% (±{np.std(cv_results['energy_saved']):.2f}%)")
    print(f"Average Data Quality: {np.mean(cv_results['data_quality']):.2f}% (±{np.std(cv_results['data_quality']):.2f}%)")
    
    # Calculate and print average pattern statistics
    print("\nPressure Pattern Analysis:")
    for category in ['true_positives', 'false_positives', 'false_negatives']:
        if category in cv_results['pattern_stats'][0]:
            print(f"\n{category.replace('_', ' ').title()}:")
            stats = cv_results['pattern_stats'][0][category]
            print(f"Mean Pressure Change: {stats['mean_change']:.3f}")
            print(f"Pressure Variability: {stats['std_change']:.3f}")
            print(f"Max Pressure Drop: {stats['max_drop']:.3f}")
            print(f"Number of Drops: {stats['n_drops']:.1f}")
            print(f"Slope: {stats['slope']:.3f}")
    
    avg_importances = np.mean(feature_importances, axis=0)
    feature_names = ['Mean Pressure Change', 'Pressure Variability', 'Number of Drops', 'Recent Trend', 'Overall Slope']
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Average Feature Importances Across Folds (Weights: {class_weights})')
    indices = np.argsort(avg_importances)[::-1]
    plt.bar(range(len(avg_importances)), avg_importances[indices])
    plt.xticks(range(len(avg_importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(f'feature_importances_weighted_{str(class_weights).replace(":", "_")}.png')
    plt.close()
    
    return cv_results

def main():
    print("Starting weighted vortex detection script...")
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = pd.read_csv('ml_ready_vortex_data.csv')
    df['pressure_diff'] = df['PRESSURE'].diff().shift(1)  # Calculate pressure difference
    X, y_detection, y_fwhm = load_and_prepare_data('ml_ready_vortex_data.csv')
    
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
    
    # Perform cross-validation for each weight scenario
    for weights in weight_scenarios:
        print(f"\nTesting with class weights: {weights}")
        results = perform_cross_validation(X, y_detection, y_fwhm, weights, df)
        all_results[str(weights)] = results
    
    # Compare results across different weights
    print("\nComparison of Results Across Different Weights:")
    metrics = ['precision', 'recall', 'f1', 'energy_saved', 'data_quality']
    for metric in metrics:
        print(f"\n{metric.capitalize()}:")
        for weights, results in all_results.items():
            mean_val = np.mean(results[metric])
            std_val = np.std(results[metric])
            print(f"{weights}: {mean_val:.3f} (±{std_val:.3f})")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main() 