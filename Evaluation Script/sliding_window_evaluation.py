#!/usr/bin/env python3
"""
Sliding Window Evaluation - Standalone Version
==============================================

Evaluates the Random Forest model on sliding windows without external imports.
"""

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy import stats

def calculate_slope(x, y):
    """Calculate linear regression slope."""
    if len(x) < 2:
        return 0.0
    try:
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    except:
        return 0.0

def engineer_features_for_window(window_data, global_mean, global_std):
    """Engineer 15 features from a 60-sample pressure window."""
    if window_data is None or len(window_data) == 0:
        return {f'feature_{i}': 0.0 for i in range(15)}
    
    # Handle both uppercase and lowercase column names
    if 'PRESSURE' in window_data.columns:
        pressure = window_data['PRESSURE'].values
    elif 'pressure' in window_data.columns:
        pressure = window_data['pressure'].values
    else:
        return {f'feature_{i}': 0.0 for i in range(15)}
    
    n_samples = len(pressure)
    
    if n_samples == 0:
        return {f'feature_{i}': 0.0 for i in range(15)}
    
    # Ensure we have at least 60 samples (pad with last value if needed)
    if n_samples < 60:
        padding = np.full(60 - n_samples, pressure[-1])
        pressure = np.concatenate([pressure, padding])
        n_samples = 60
    
    # Calculate indices for first and second half
    mid_point = n_samples // 2
    first_half = pressure[:mid_point]
    second_half = pressure[mid_point:]
    
    # Feature 1: Overall slope
    x = np.arange(n_samples)
    overall_slope = calculate_slope(x, pressure)
    
    # Feature 2: First half slope
    x_first = np.arange(len(first_half))
    first_half_slope = calculate_slope(x_first, first_half)
    
    # Feature 3: Second half slope
    x_second = np.arange(len(second_half))
    second_half_slope = calculate_slope(x_second, second_half)
    
    # Feature 4: Trend consistency
    if len(pressure) >= 4:
        window_size = min(10, len(pressure) // 3)
        slopes = []
        for i in range(len(pressure) - window_size):
            x = np.arange(window_size)
            y = pressure[i:i + window_size]
            slope = calculate_slope(x, y)
            slopes.append(slope)
        slope_std = np.std(slopes) if slopes else 0
        trend_consistency = 1.0 / (1.0 + slope_std) if slope_std > 0 else 1.0
    else:
        trend_consistency = 0.0
    
    # Feature 5: Pressure drop
    pressure_drop = np.max(pressure) - np.min(pressure)
    
    # Feature 6: Drop rate
    max_drop = 0.0
    for i in range(len(pressure) - 1):
        drop = pressure[i] - pressure[i + 1]
        max_drop = max(max_drop, drop)
    drop_rate = max_drop
    
    # Feature 7: Minimum position
    min_idx = np.argmin(pressure)
    min_position = min_idx / (len(pressure) - 1) if len(pressure) > 1 else 0.5
    
    # Feature 8: Mean
    mean = np.mean(pressure)
    
    # Feature 9: Standard deviation
    std = np.std(pressure)
    
    # Feature 10: Range
    range_val = np.max(pressure) - np.min(pressure)
    
    # Feature 11: First half mean
    first_half_mean = np.mean(first_half)
    
    # Feature 12: Second half mean
    second_half_mean = np.mean(second_half)
    
    # Feature 13: Mean ratio
    mean_ratio = second_half_mean / first_half_mean if first_half_mean != 0 else 1.0
    
    # Feature 14: Minimum z-score
    min_pressure = np.min(pressure)
    if global_std > 0:
        min_zscore = (min_pressure - global_mean) / global_std
    else:
        min_zscore = 0.0
    
    # Feature 15: Anomaly strength
    if len(pressure) >= 3:
        x = np.arange(len(pressure))
        slope, intercept, _, _, _ = stats.linregress(x, pressure)
        min_idx = np.argmin(pressure)
        expected_pressure = slope * min_idx + intercept
        actual_pressure = pressure[min_idx]
        anomaly_strength = abs(actual_pressure - expected_pressure)
    else:
        anomaly_strength = 0.0
    
    features = {
        'overall_slope': overall_slope,
        'first_half_slope': first_half_slope,
        'second_half_slope': second_half_slope,
        'trend_consistency': trend_consistency,
        'pressure_drop': pressure_drop,
        'drop_rate': drop_rate,
        'min_position': min_position,
        'mean': mean,
        'std': std,
        'range': range_val,
        'first_half_mean': first_half_mean,
        'second_half_mean': second_half_mean,
        'mean_ratio': mean_ratio,
        'min_zscore': min_zscore,
        'anomaly_strength': anomaly_strength
    }
    
    return features

def load_and_process_sliding_windows(split, step_size=10):
    """Load and process sliding windows for evaluation."""
    print(f"Loading {split} sliding windows...")
    
    # Load sliding windows
    input_file = f"{split}_sliding_windows_step{step_size}.csv"
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} sliding windows")
    
    # Show label distribution
    label_counts = df['label'].value_counts()
    print("  Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"    {label}: {count:,} ({percentage:.1f}%)")
    
    return df

def engineer_features_from_sliding_windows(df, sample_size=5000):
    """Engineer features from sliding windows."""
    print("Engineering features from sliding windows...")
    
    # Calculate global stats
    print("  Calculating global statistics...")
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    all_pressures = []
    for _, row in sample_df.iterrows():
        try:
            window_data = pd.read_json(row['window_data'], orient='records')
            if 'PRESSURE' in window_data.columns:
                all_pressures.extend(window_data['PRESSURE'].values)
        except:
            continue
    
    global_mean = np.mean(all_pressures) if all_pressures else 745.0
    global_std = np.std(all_pressures) if all_pressures else 8.0
    print(f"    Global mean: {global_mean:.3f}")
    print(f"    Global std: {global_std:.3f}")
    
    # Engineer features for all windows
    all_features = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing windows"):
        try:
            window_data = pd.read_json(row['window_data'], orient='records')
            features = engineer_features_for_window(window_data, global_mean, global_std)
            
            feature_row = {
                'window_id': row['window_id'],
                'label': row['label'],
                **features
            }
            all_features.append(feature_row)
        except:
            continue
    
    features_df = pd.DataFrame(all_features)
    print(f"  Generated {len(features_df):,} feature vectors")
    
    return features_df

def train_model(train_features_df):
    """Train Random Forest model on training features."""
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    
    # Prepare training data
    feature_cols = [col for col in train_features_df.columns if col not in ['window_id', 'label', 'event_sclk', 'split']]
    X_train = train_features_df[feature_cols].values
    y_train = train_features_df['label'].values
    
    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Class distribution: {np.bincount(y_train)}")
    print(f"Features: {feature_cols}")
    
    # Train model
    print("\nTraining Random Forest...")
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
    print("Training completed!")
    print("=" * 70)
    
    return rf_model, feature_cols

def evaluate_model(rf_model, feature_cols, eval_features_df, split_name):
    """Evaluate trained model on sliding windows."""
    print(f"\nEvaluating on {split_name.upper()} sliding windows...")
    
    # Prepare evaluation data
    print(f"  Preparing {split_name} evaluation data...")
    
    # Filter out 'Omit' labels
    valid_eval_df = eval_features_df[eval_features_df['label'] != 'Omit'].copy()
    print(f"    After filtering 'Omit': {len(valid_eval_df):,} windows")
    
    # Convert to binary labels
    valid_eval_df['label'] = valid_eval_df['label'].map({'True': 1, 'False': 0})
    
    X_eval = valid_eval_df[feature_cols].values
    y_eval = valid_eval_df['label'].values
    
    print(f"    Evaluation data: {X_eval.shape[0]} samples")
    print(f"    Evaluation class distribution: {np.bincount(y_eval)}")
    
    # Make predictions
    print("  Making predictions...")
    y_pred = rf_model.predict(X_eval)
    y_proba = rf_model.predict_proba(X_eval)[:, 1]
    
    # Calculate metrics
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ON {split_name.upper()} SLIDING WINDOWS")
    print(f"{'='*70}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_eval, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"True Negative   {tn:<20} {fp}")
    print(f"True Positive   {fn:<20} {tp}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_eval, y_pred, target_names=['Negative', 'Positive']))
    
    # Detailed Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    roc_auc = roc_auc_score(y_eval, y_proba)
    
    print(f"\nDetailed Metrics (Positive Class):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    
    print(f"\nError Analysis:")
    print(f"  False Positives: {fp} (incorrectly predicted as vortex)")
    print(f"  False Negatives: {fn} (missed vortex detections)")
    print(f"  True Positives:  {tp} (correctly detected vortices)")
    print(f"  True Negatives:  {tn} (correctly identified non-vortex)")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

def main():
    """Main execution function."""
    print("=" * 70)
    print("SLIDING WINDOW EVALUATION - IMPROVED VERSION")
    print("=" * 70)
    print("Train once, evaluate multiple times for efficiency")
    print("=" * 70)
    
    # STEP 1: Load training features (from fixed windows)
    print("\nSTEP 1: Loading training features...")
    train_features_df = pd.read_csv("datasets/train_features.csv")
    print(f"  Loaded {len(train_features_df):,} training feature vectors")
    
    # STEP 2: Train model ONCE
    print("\nSTEP 2: Training Random Forest model...")
    rf_model, feature_cols = train_model(train_features_df)
    
    # STEP 3: Load pre-computed validation features
    print("\nSTEP 3: Loading pre-computed validation features...")
    val_features_df = pd.read_csv("datasets/val_sliding_features.csv")
    print(f"  Loaded {len(val_features_df):,} validation feature vectors")
    
    # STEP 4: Evaluate on validation
    print("\nSTEP 4: Evaluating on validation...")
    val_metrics = evaluate_model(rf_model, feature_cols, val_features_df, "validation")
    
    print("\n" + "=" * 70)
    print("VALIDATION EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Performance on Validation Sliding Windows:")
    print(f"  F1-Score:  {val_metrics['f1_score']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall:    {val_metrics['recall']:.4f}")
    print(f"  ROC AUC:   {val_metrics['roc_auc']:.4f}")
    
    # STEP 5: Load pre-computed test features
    print("\nSTEP 5: Loading pre-computed test features...")
    test_features_df = pd.read_csv("datasets/test_sliding_features.csv")
    print(f"  Loaded {len(test_features_df):,} test feature vectors")
    
    # STEP 6: Evaluate on test (SAME MODEL, no retraining!)
    print("\nSTEP 6: Evaluating on test (using same trained model)...")
    test_metrics = evaluate_model(rf_model, feature_cols, test_features_df, "test")
    
    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Performance on Test Sliding Windows:")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    
    print("\n" + "=" * 70)
    print("SLIDING WINDOW EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    # Summary comparison
    print(f"\nSUMMARY - SLIDING WINDOW PERFORMANCE (Same Model on Both):")
    print(f"{'Metric':<12} {'Validation':<12} {'Test':<12}")
    print(f"{'F1-Score':<12} {val_metrics['f1_score']:.4f}       {test_metrics['f1_score']:.4f}")
    print(f"{'Precision':<12} {val_metrics['precision']:.4f}       {test_metrics['precision']:.4f}")
    print(f"{'Recall':<12} {val_metrics['recall']:.4f}       {test_metrics['recall']:.4f}")
    print(f"{'ROC AUC':<12} {val_metrics['roc_auc']:.4f}       {test_metrics['roc_auc']:.4f}")
    
    print(f"\nNote: Model trained ONCE on {len(train_features_df)} fixed windows,")
    print(f"      then evaluated on {len(val_features_df):,} validation and {len(test_features_df):,} test sliding windows")

if __name__ == "__main__":
    main()
