import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import argparse
from pathlib import Path
from joblib import dump, load
from feature_processor import FeatureProcessor
from typing import Tuple

# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
# from visualize_metrics import visualize_model_metrics, create_model_report
# from analyze_features import analyze_feature_importance

# Define feature names at module level
FEATURE_NAMES = [
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

# Define default model path
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'vortex_model.joblib'

def prepare_features(data: pd.DataFrame, data_dir: str, force_recalculate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features using the FeatureProcessor."""
    try:
        processor = FeatureProcessor(data_dir)
        X = processor.prepare_features(data, force_recalculate)
        print("Features prepared successfully")
        
        # Generate labels looking ahead 10 samples, matching backup model
        y = np.zeros(len(X))
        for i in range(len(X)):
            if i + 10 <= len(data):
                y[i] = 1 if any(data['gt_detection_win'].iloc[i:i+10] == 1) else 0
        print("Labels generated successfully")
        return X, y
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, data_dir: str) -> RandomForestClassifier:
    """Train the model with balanced class weights."""
    print("\nTraining model...")
    
    try:
        # Calculate class weights matching backup model
        n_samples = len(y_train)
        n_positives = np.sum(y_train == 1)
        n_negatives = n_samples - n_positives
        class_weights = {
            0: 1.0,
            1: n_negatives / n_positives  # Give more weight to minority class
        }
        
        # Initialize model with calculated class weights
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        model_path = Path(data_dir) / 'models' / 'vortex_model.joblib'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(model, model_path)
        print(f"\nModel saved to: {model_path}")
        
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        raise

def analyze_predictions(model, X_test, y_test):
    """Analyze model predictions in detail and save comprehensive metrics"""
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    # Analyze feature importance
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(FEATURE_NAMES, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance Analysis:")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")
    
    # Analyze probability distributions
    print("\nProbability Distribution Analysis:")
    true_vortex_probs = y_pred_proba[y_test == 1][:, 1]
    non_vortex_probs = y_pred_proba[y_test == 0][:, 1]
    
    print("\nTrue Vortex Probability Stats:")
    print(f"Mean probability: {np.mean(true_vortex_probs):.4f}")
    print(f"Median probability: {np.median(true_vortex_probs):.4f}")
    print(f"Std deviation: {np.std(true_vortex_probs):.4f}")
    print(f"Min probability: {np.min(true_vortex_probs):.4f}")
    print(f"Max probability: {np.max(true_vortex_probs):.4f}")
    print(f"Probability distribution:")
    print(f"  0-0.2: {np.sum((true_vortex_probs >= 0.0) & (true_vortex_probs < 0.2))}")
    print(f"  0.2-0.4: {np.sum((true_vortex_probs >= 0.2) & (true_vortex_probs < 0.4))}")
    print(f"  0.4-0.6: {np.sum((true_vortex_probs >= 0.4) & (true_vortex_probs < 0.6))}")
    print(f"  0.6-0.8: {np.sum((true_vortex_probs >= 0.6) & (true_vortex_probs < 0.8))}")
    print(f"  0.8-1.0: {np.sum((true_vortex_probs >= 0.8) & (true_vortex_probs <= 1.0))}")
    
    # Calculate optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\nOptimal threshold (maximizing F1-score): {optimal_threshold:.4f}")
    
    # Set threshold for classification
    threshold = optimal_threshold
    print(f"Using optimal threshold: {threshold:.4f}")
    print(f"Number of probabilities above threshold: {np.sum(y_pred_proba[:, 1] >= threshold)}")
    
    # Make predictions with optimal threshold
    y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
    
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
    parser = argparse.ArgumentParser(description='Train or load vortex prediction model')
    parser.add_argument('--data-fraction', type=float, default=1.0,
                      help='Fraction of data to use for training (0.0 to 1.0). Default is 1.0 (100%)')
    parser.add_argument('--model-path', type=str, default=str(DEFAULT_MODEL_PATH),
                      help='Path to load existing model. Defaults to standard model location.')
    parser.add_argument('--force-recalculate', action='store_true',
                      help='Force recalculation of features')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to the data file. If not provided, will look in data/ml_ready_vortex_data.csv')
    parser.add_argument('--force-retrain', action='store_true',
                      help='Force retraining of the model even if it exists')
    args = parser.parse_args()
    
    print("Starting vortex prediction model...")
    
    try:
        # Load data
        print("Loading data...")
        start_time = time.time()
        
        if args.data_path:
            data_path = Path(args.data_path)
        else:
            data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
        print(f"Full dataset size: {len(df)} samples")
        
        # Ensure minimum test set size (51 samples for feature calculation)
        MIN_TEST_SIZE = 51
        if len(df) <= MIN_TEST_SIZE:
            raise ValueError(f"Dataset must have more than {MIN_TEST_SIZE} samples for feature calculation")
        
        # Calculate train size based on data fraction
        if args.data_fraction >= 1.0:
            # Use 80/20 split when using all data
            print("Using 80% of data for training and 20% for testing...")
            train_size = int(len(df) * 0.8)
        else:
            train_size = int(len(df) * args.data_fraction)
            if len(df) - train_size < MIN_TEST_SIZE:
                print(f"Warning: Requested split would leave less than {MIN_TEST_SIZE} samples for testing.")
                print(f"Adjusting to use {len(df) - MIN_TEST_SIZE} samples for training...")
                train_size = len(df) - MIN_TEST_SIZE
            else:
                print(f"Using {args.data_fraction*100:.1f}% of data for training...")
        
        # Create FeatureProcessor and calculate features on full dataset
        print("Preparing features for full dataset...")
        all_features = prepare_features(df, str(Path(__file__).parent.parent), args.force_recalculate)
        
        # Split features and labels (accounting for window size)
        window_size = 50  # Window size for feature calculation
        train_end = train_size - window_size
        X_train = all_features[0][:train_end]
        X_test = all_features[0][train_end:]
        y_train = all_features[1][:train_end]  # Use train_end to align with features
        y_test = all_features[1][train_end:]  # Test labels start at train_end
        
        # Calculate class weights matching backup model
        n_samples = len(y_train)
        n_positives = np.sum(y_train == 1)
        n_negatives = n_samples - n_positives
        class_weights = {
            0: 1.0,
            1: n_negatives / n_positives  # Give more weight to minority class
        }
        print(f"\nClass weights: {class_weights}")
        
        # Validate lengths match
        if len(X_train) != len(y_train):
            raise ValueError(f"Feature and label lengths don't match for training: X_train={len(X_train)}, y_train={len(y_train)}")
        if len(X_test) != len(y_test):
            raise ValueError(f"Feature and label lengths don't match for testing: X_test={len(X_test)}, y_test={len(y_test)}")
        
        print(f"Training set size: {len(X_train)} samples")
        print(f"Test set size: {len(X_test)} samples")
        
        # Train or load model
        model_path = Path(args.model_path)
        if model_path.exists() and not args.force_retrain:
            print(f"\nLoading existing model from: {model_path}")
            model = load(model_path)
            print("\nUsing existing model for evaluation")
        else:
            if args.force_retrain:
                print("\nForce retrain flag set. Training new model...")
            else:
                print(f"\nNo model found at {model_path}. Training new model...")
            model = train_model(X_train, y_train, 'models/prediction_model')
            
        # Analyze feature values
        print("\nFeature Analysis:")
        print("Training set feature statistics:")
        print(f"Mean: {np.mean(X_train, axis=0)}")
        print(f"Std: {np.std(X_train, axis=0)}")
        print(f"Min: {np.min(X_train, axis=0)}")
        print(f"Max: {np.max(X_train, axis=0)}")
        
        # Analyze feature values for positive vs negative cases
        pos_features = X_train[y_train == 1]
        neg_features = X_train[y_train == 0]
        print("\nFeature statistics for positive cases:")
        print(f"Mean: {np.mean(pos_features, axis=0)}")
        print(f"Std: {np.std(pos_features, axis=0)}")
        print("\nFeature statistics for negative cases:")
        print(f"Mean: {np.mean(neg_features, axis=0)}")
        print(f"Std: {np.std(neg_features, axis=0)}")
        
        # Analyze model
        analyze_predictions(model, X_test, y_test)
        
    except Exception as e:
        print(f"\nError during model training/evaluation: {str(e)}")
        raise
        
    print("\nModel analysis complete!")

if __name__ == "__main__":
    main() 