import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
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
from visualize_metrics import visualize_model_metrics, create_model_report
from analyze_features import analyze_feature_importance

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

def prepare_features(data: pd.DataFrame, data_dir: str, force_recalculate: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features using the FeatureProcessor."""
    try:
        processor = FeatureProcessor(data_dir)
        X = processor.prepare_features(data, force_recalculate)
        y = data['gt_detection_win'].values[50:]  # Skip first 50 samples due to window size
        return X, y
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, data_dir: str) -> RandomForestClassifier:
    """Train the model with balanced class weights."""
    print("\nTraining model...")
    
    try:
        # Initialize model with balanced class weights
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
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
    
    # Make predictions with custom threshold
    y_pred_proba = model.predict_proba(X_test)
    y_pred = (y_pred_proba[:, 1] >= 0.150).astype(int)
    
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
    parser.add_argument('--data-fraction', type=float, default=0.33,
                      help='Fraction of data to use for training (0.0 to 1.0)')
    parser.add_argument('--model-path', type=str, help='Path to load existing model')
    parser.add_argument('--force-recalculate', action='store_true',
                      help='Force recalculation of features')
    parser.add_argument('--data-path', type=str, default=None,
                      help='Path to the data file. If not provided, will look in data/ml_ready_vortex_data.csv')
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
        
        # Handle data splitting
        if args.data_fraction >= 1.0:
            print("Using 80% of data for training and 20% for testing...")
            train_size = int(len(df) * 0.8)  # Use 80% for training
        else:
            print(f"Using {args.data_fraction*100:.1f}% of data for training...")
            train_size = int(len(df) * args.data_fraction)
        
        # Split data
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        
        print(f"Training set size: {len(df_train)} samples")
        print(f"Test set size: {len(df_test)} samples")
        
        # Prepare features for training data
        print("Preparing features for training data...")
        X_train, y_train = prepare_features(df_train, '../', args.force_recalculate)
        
        # Prepare features for test data
        print("Preparing features for test data...")
        X_test, y_test = prepare_features(df_test, '../', args.force_recalculate)
        
        # Train or load model
        if args.model_path:
            print(f"\nModel loaded from: {args.model_path}")
            model = load(args.model_path)
            print("\nUsing existing model for evaluation")
        else:
            model = train_model(X_train, y_train, 'models/prediction_model')
        
        # Analyze model
        analyze_predictions(model, X_test, y_test)
        
    except Exception as e:
        print(f"\nError during model training/evaluation: {str(e)}")
        raise
        
    print("\nModel analysis complete!")

if __name__ == "__main__":
    main() 