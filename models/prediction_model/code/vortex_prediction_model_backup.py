import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
from pathlib import Path

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from visualize_metrics import visualize_model_metrics, create_model_report
from analyze_features import analyze_feature_importance

def prepare_features(df, window_size=50, random_state=42):
    """Prepare features from pressure data for vortex prediction using a sliding window approach"""
    print("Preparing features...")
    
    # Set numpy random seed for reproducibility
    np.random.seed(random_state)
    
    # Calculate pressure difference
    df['Pressure_Difference'] = df['PRESSURE'] - df['PRESSURE_MA_500']
    
    # Create features from pressure patterns
    features = []
    labels = []
    
    # Initialize the first window
    current_window = df['Pressure_Difference'].iloc[0:window_size].values
    
    # Slide the window through the data
    for i in range(window_size, len(df)):
        # Update window by removing oldest sample and adding newest
        # Shift values right by 1 (oldest values stay on left)
        current_window = np.roll(current_window, 1)
        current_window[0] = df['Pressure_Difference'].iloc[i]
        
        # Calculate feature statistics on the current window
        features.append([
            np.mean(current_window),  # Mean pressure
            np.std(current_window),   # Pressure variability
            np.min(current_window),   # Minimum pressure
            np.max(current_window),   # Maximum pressure
            np.mean(np.diff(current_window)),  # Average rate of change
            np.std(np.diff(current_window)),   # Rate of change variability
            np.mean(current_window[-10:]),     # Recent pressure mean
            np.std(current_window[-10:]),      # Recent pressure variability
            np.mean(np.diff(current_window[-10:])),  # Recent rate of change
            np.std(np.diff(current_window[-10:]))    # Recent rate of change variability
        ])
        
        # Label is 1 if a detection window starts in the next 10 samples
        # Using gt_detection_win instead of gt_fwhm to match the detection window approach
        label = 1 if any(df['gt_detection_win'].iloc[i:i+10] == 1) else 0
        labels.append(label)
    
    return np.array(features), np.array(labels)

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train the vortex prediction model using a sliding window approach.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Ground truth labels
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Calculate class weights
    n_samples = len(y_train)
    n_positive = sum(y_train == 1)
    n_negative = n_samples - n_positive
    class_weight = {
        0: 1.0,
        1: n_negative / n_positive  # Give more weight to minority class
    }
    print(f"\nClass distribution in training set:")
    print(f"Class 0 (no vortex): {n_negative} samples")
    print(f"Class 1 (vortex): {n_positive} samples")
    print(f"Class weights: {class_weight}")
    
    # Train model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    # Make predictions with custom threshold
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= 0.150).astype(int)
    
    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Visualize model metrics
        visualize_model_metrics(
            model=model,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            model_name='Vortex Prediction Model',
            save_dir=results_dir
        )
        
        # Create HTML report
        create_model_report('Vortex Prediction Model', results_dir)
        
        # Analyze feature importance
        feature_names = [
            'Recent Pressure Mean',             # Mean of last 10 samples
            'Recent Rate of Change',            # Mean rate of change in last 10 samples
            'Pressure Variability',             # Standard deviation of pressure
            'Average Rate of Change',           # Mean rate of change across window
            'Recent Pressure Variability',      # Standard deviation of last 10 samples
            'Rate of Change Variability',       # Standard deviation of rate of change
            'Recent Rate of Change Variability',# Standard deviation of rate of change in last 10 samples
            'Mean Pressure',                    # Mean pressure across window
            'Min Pressure',                     # Minimum pressure in window
            'Max Pressure'                      # Maximum pressure in window
        ]
        
        analyze_feature_importance(
            model=model,
            feature_names=feature_names,
            X_test=X_test,
            y_test=y_test,
            save_dir=results_dir / 'feature_analysis',
            model_name='Vortex Prediction Model'
        )
    except Exception as e:
        print(f"\nWarning: Error during analysis/visualization: {str(e)}")
        print("Continuing with model training...")
    
    return model, X_test, y_test

def analyze_predictions(model, X_test, y_test):
    """Analyze model predictions in detail and save comprehensive metrics"""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Make predictions with custom threshold
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= 0.150).astype(int)
    
    # Calculate prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate basic metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate energy efficiency metrics
    total_samples = len(y_test)
    true_vortices = sum(y_test == 1)
    predicted_vortices = sum(y_pred == 1)
    true_positives = sum((y_pred == 1) & (y_test == 1))
    false_positives = sum((y_pred == 1) & (y_test == 0))
    false_negatives = sum((y_pred == 0) & (y_test == 1))
    
    # Energy efficiency metrics
    energy_saved = (total_samples - predicted_vortices) / total_samples  # % of time in low energy mode
    data_quality = true_positives / true_vortices  # % of vortices caught
    
    # Print comprehensive analysis
    print("\nComprehensive Model Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"True vortices: {true_vortices}")
    print(f"Predicted vortices: {predicted_vortices}")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print("\nPerformance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nEnergy Efficiency Metrics:")
    print(f"Energy Saved: {energy_saved:.2%}")
    print(f"Data Quality: {data_quality:.2%}")
    
    # Analyze prediction probabilities
    if false_positives > 0:
        print("\nFalse Positive Analysis:")
        print(f"Average probability for false positives: {np.mean(y_pred_proba[(y_pred == 1) & (y_test == 0), 1]):.4f}")
    
    if false_negatives > 0:
        print("\nFalse Negative Analysis:")
        print(f"Average probability for false negatives: {np.mean(y_pred_proba[(y_pred == 0) & (y_test == 1), 1]):.4f}")
    
    # Save results to file
    results = {
        'total_samples': total_samples,
        'true_vortices': true_vortices,
        'predicted_vortices': predicted_vortices,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'energy_saved': energy_saved,
        'data_quality': data_quality
    }
    
    # Add probability metrics if available
    if false_positives > 0:
        results['fp_avg_prob'] = np.mean(y_pred_proba[(y_pred == 1) & (y_test == 0), 1])
    if false_negatives > 0:
        results['fn_avg_prob'] = np.mean(y_pred_proba[(y_pred == 0) & (y_test == 1), 1])
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_dir / 'model_metrics.csv', index=False)
    
    # Save detailed classification report
    with open(results_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    return results

def main():
    print("Starting vortex prediction model training...")
    
    # Set random seed for reproducibility
    random_state = 42
    
    # Load data
    print("Loading data...")
    start_time = time.time()
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    df = pd.read_csv(data_path)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # TEMPORARY: Use only first 1/3rd of data for testing
    # TODO: Remove this line and use full dataset in production
    df = df.iloc[:len(df)//3]
    print(f"Using {len(df)} samples for testing (1/3rd of total data)")
    
    # Prepare features and labels
    X, y = prepare_features(df, random_state=random_state)
    print(f"Prepared {len(X)} samples with {X.shape[1]} features")
    
    # Train model and get results
    model, X_test, y_test = train_model(X, y, random_state=random_state)
    
    # Analyze predictions
    analyze_predictions(model, X_test, y_test)
    
    print("\nModel training and analysis complete!")

if __name__ == "__main__":
    main() 