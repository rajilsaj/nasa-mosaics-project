import pandas as pd
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from pathlib import Path
import time
import argparse
import matplotlib.pyplot as plt
import joblib
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import gc
print("Using GPU:", tf.config.list_physical_devices('GPU'))
mixed_precision.set_global_policy("mixed_float16")



# Add utils directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'utils'))
from visualize_lstm_metrics import visualize_lstm_metrics, create_lstm_report

class VortexLSTMModel:
    """Single-stage LSTM model for vortex prediction."""
    
    def __init__(self, window_size: int = 60, prediction_threshold: float = 0.2):
        """Initialize the LSTM model."""
        self.window_size = window_size
        self.prediction_threshold = prediction_threshold
        self.model = None
        
    def prepare_sequences(self, data: pd.DataFrame, apply_sampling: bool = True) -> tuple:
        """Prepare sequences for vortex prediction.
        
        Args:
            data: Input data
            apply_sampling: If True, creates balanced dataset for training/validation.
                          If False, processes all available data for testing/evaluation.
        """
        pressure_values = data['PRESSURE'].values
        gt_detection = data['gt_detection_win'].values
        gt_fwhm = data['gt_fwhm'].values
        
        # Normalize pressure values
        pressure_mean = np.mean(pressure_values)
        pressure_std = np.std(pressure_values)
        normalized_pressure = (pressure_values - pressure_mean) / pressure_std
        
        # Calculate rate of change (pressure differences)
        rate_of_change = np.diff(pressure_values)
        rate_of_change = np.concatenate([[0], rate_of_change])
        
        # Normalize rate of change
        roc_mean = np.mean(rate_of_change)
        roc_std = np.std(rate_of_change)
        normalized_roc = (rate_of_change - roc_mean) / roc_std
        
        print("\nNormalization statistics:")
        print(f"Pressure - Mean: {pressure_mean:.2f}, Std: {pressure_std:.2f}")
        print(f"Rate of Change - Mean: {roc_mean:.2f}, Std: {roc_std:.2f}")
        
        if not apply_sampling:
            # For test/eval: process all available data
            print("\nProcessing all available data for test/evaluation...")
            sequences_list = []
            labels_list = []
            
            # Create sequences for every possible window
            for i in range(self.window_size, len(data) - self.window_size + 1):
                # Get the window
                pressure_window = normalized_pressure[i:i+self.window_size]
                roc_window = normalized_roc[i:i+self.window_size]
                
                # Create sequence
                sequence = np.zeros((self.window_size, 2))
                sequence[:, 0] = pressure_window
                sequence[:, 1] = roc_window
                
                # Get label (1 if any point in the window is a vortex)
                label = 1 if np.any(np.logical_or(gt_detection[i:i+self.window_size] == 1, 
                                                gt_fwhm[i:i+self.window_size] == 1)) else 0
                
                sequences_list.append(sequence)
                labels_list.append(label)
            
            sequences = np.array(sequences_list)
            labels = np.array(labels_list)
            
            print("\nFull data statistics:")
            print(f"Total sequences: {len(sequences)}")
            print(f"Vortex sequences: {sum(labels)}")
            print(f"Non-vortex sequences: {len(labels) - sum(labels)}")
            print(f"Ratio: {(len(labels) - sum(labels)) / sum(labels):.2f}:1")
            
            return sequences, labels
        
        # For training/validation: create balanced dataset
        # Combine both ground truth conditions - vortex is true if either is 1
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1).astype(int)
        
        # Find all vortex events
        vortex_indices = np.where(gt_combined == 1)[0]
        print(f"\nFound {len(vortex_indices)} vortex events")
        
        # Debug: Print first few vortex indices and their values
        print("\nFirst few vortex events (normalized):")
        for i in range(min(5, len(vortex_indices))):
            idx = vortex_indices[i]
            print(f"Index {idx}:")
            print(f"  Normalized Pressure: {normalized_pressure[idx]:.4f}")
            print(f"  Normalized ROC: {normalized_roc[idx]:.4f}")
            print(f"  gt_detection_win: {gt_detection[idx]}")
            print(f"  gt_fwhm: {gt_fwhm[idx]}")
            print(f"  Combined: {gt_combined[idx]}")
        
        sequences_list = []
        labels_list = []
        
        # Process each vortex event
        for i, vortex_idx in enumerate(vortex_indices):
            if i % 1000 == 0:
                print(f"Processing vortex event {i}/{len(vortex_indices)}")
            
            # Skip if we don't have enough data before the vortex
            if vortex_idx < self.window_size:
                continue
                
            # Get non-vortex window before the vortex
            non_vortex_start = vortex_idx - self.window_size
            non_vortex_end = vortex_idx
            
            # Get vortex window
            vortex_start = vortex_idx
            vortex_end = vortex_idx + self.window_size
            
            # Skip if we don't have enough data after the vortex
            if vortex_end > len(data):
                continue
            
            # Create non-vortex sequence
            non_vortex_pressure = normalized_pressure[non_vortex_start:non_vortex_end]
            non_vortex_roc = normalized_roc[non_vortex_start:non_vortex_end]
            non_vortex_sequence = np.zeros((self.window_size, 2))
            non_vortex_sequence[:, 0] = non_vortex_pressure
            non_vortex_sequence[:, 1] = non_vortex_roc
            
            # Create vortex sequence
            vortex_pressure = normalized_pressure[vortex_start:vortex_end]
            vortex_roc = normalized_roc[vortex_start:vortex_end]
            vortex_sequence = np.zeros((self.window_size, 2))
            vortex_sequence[:, 0] = vortex_pressure
            vortex_sequence[:, 1] = vortex_roc
            
            # Debug: Print first few sequences
            if i < 2:
                print(f"\nSequence {i}:")
                print("Non-vortex window (normalized):")
                print(f"  Start idx: {non_vortex_start}, End idx: {non_vortex_end}")
                print(f"  Pressure values: {non_vortex_pressure[:5]}...")
                print(f"  ROC values: {non_vortex_roc[:5]}...")
                print("\nVortex window (normalized):")
                print(f"  Start idx: {vortex_start}, End idx: {vortex_end}")
                print(f"  Pressure values: {vortex_pressure[:5]}...")
                print(f"  ROC values: {vortex_roc[:5]}...")
            
            # Add both sequences
            sequences_list.extend([non_vortex_sequence, vortex_sequence])
            labels_list.extend([0, 1])
        
        # Convert to numpy arrays
        sequences = np.array(sequences_list)
        labels = np.array(labels_list)
        
        print("\nFinal sequence statistics:")
        print(f"Total sequences: {len(sequences)}")
        print(f"Vortex sequences: {sum(labels)}")
        print(f"Non-vortex sequences: {len(labels) - sum(labels)}")
        print(f"Final ratio: {(len(labels) - sum(labels)) / sum(labels):.2f}:1")
        
        return sequences, labels
        
    def focal_loss(self, gamma=2.0, alpha=0.95):
        def focal_loss_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            
            loss_1 = -alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
            loss_0 = -(1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
            
            return tf.reduce_mean(loss_1 + loss_0)
        return focal_loss_fixed
    
    def calculate_alpha(self, y_train: np.ndarray) -> float:
        """Calculate alpha for focal loss based on class distribution."""
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        total = n_positive + n_negative
        
        # Alpha is the inverse of the positive class frequency
        # This gives higher weight to the positive class (vortices)
        alpha = n_negative / total
        
        print(f"\nClass distribution for alpha calculation:")
        print(f"Positive examples (vortices): {n_positive}")
        print(f"Negative examples (non-vortices): {n_negative}")
        print(f"Calculated alpha: {alpha:.4f}")
        
        return alpha
    
    def build_model(self, input_shape: tuple, alpha: float = None):
        """Build the vortex prediction model."""
        model = Sequential([
            LSTM(128,  # More units in single layer
                 kernel_regularizer=l2(0.0005),  # Reduced regularization
                 recurrent_regularizer=l2(0.0005),
                 input_shape=input_shape),
            Dense(32, activation='relu'),
            Dropout(0.2),  # Reduced dropout
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.focal_loss(gamma=2.0, alpha=alpha),
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(curve='ROC', name='roc_auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc')]
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256):
        """Train the model."""
        # Print statistics about the data
        print("\nTraining Data Statistics:")
        print(f"Total examples: {len(X_train)}")
        print(f"Vortex examples: {sum(y_train)}")
        print(f"Non-vortex examples: {len(y_train) - sum(y_train)}")
        print(f"Ratio: {(len(y_train) - sum(y_train)) / sum(y_train):.2f}:1")
        
        # Add detailed debugging
        print("\nVerifying sampling results:")
        vortex_indices = np.where(y_train == 1)[0]
        non_vortex_indices = np.where(y_train == 0)[0]
        print(f"Number of vortex sequences: {len(vortex_indices)}")
        print(f"Number of non-vortex sequences: {len(non_vortex_indices)}")
        print(f"First few vortex indices: {vortex_indices[:10]}")
        
        # Check distribution in first few sequences
        print("\nChecking first few sequences:")
        for i in range(min(20, len(y_train))):
            print(f"Sequence {i}: Label = {y_train[i]}")
        
        print("\nValidation Data Statistics:")
        print(f"Total examples: {len(X_val)}")
        print(f"Vortex examples: {sum(y_val)}")
        print(f"Non-vortex examples: {len(y_val) - sum(y_val)}")
        print(f"Ratio: {(len(y_val) - sum(y_val)) / sum(y_val):.2f}:1")
        
        # Add logging for class weights
        print("\nVerifying class weights in training batches...")
        for i in range(0, len(X_train), batch_size):
            batch_y = y_train[i:i+batch_size]
            pos_weight = np.sum(batch_y == 1) / len(batch_y)
            neg_weight = np.sum(batch_y == 0) / len(batch_y)
            print(f"Batch {i//batch_size}: Positive weight: {pos_weight:.4f}, Negative weight: {neg_weight:.4f}")
            if i > batch_size * 5:  # Only print first 5 batches
                break
        
        # Calculate alpha based on class distribution
        alpha = self.calculate_alpha(y_train)
        print(f"Alpha: {alpha}")
        
        # Train model
        print("\nTraining vortex prediction model...")
        self.model = self.build_model((self.window_size, 2), alpha=alpha)
        
        # Add learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            mode='max',
            verbose=1
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                EarlyStopping(monitor='val_pr_auc', patience=5, mode='max'),
                ModelCheckpoint(
                    'best_model.h5',
                    monitor='val_pr_auc',
                    save_best_only=True,
                    mode='max'
                ),
                reduce_lr
            ]
        )
        
        return history
    
    def predict(self, X):
        """Make predictions using the model."""
        return self.model.predict(X).flatten()
    
    def predict_real_time(self, pressure_readings: np.ndarray) -> float:
        """Make real-time predictions on new pressure readings."""
        if len(pressure_readings) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} pressure readings")
            
        current_readings = pressure_readings[-self.window_size:]
        sequence = current_readings.reshape(1, self.window_size, 2)
        return self.model.predict(sequence)[0][0]
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        
        y_pred_proba = self.predict(X_test)
        
        # Try different thresholds to find the best F1 score
        best_f1 = 0
        best_threshold = 0.5
        thresholds = np.linspace(0.3, 0.7, 41)
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        print(f"\nBest threshold: {best_threshold:.4f}")
        print(f"Best F1 score: {best_f1:.4f}")
        
        # Use the best threshold for final evaluation
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'threshold': best_threshold
        }

def find_detection_windows(data: pd.DataFrame) -> list:
    """Find all detection windows (where gt_detection_win == 1)."""
    windows = []
    in_window = False
    start_idx = None
    
    for idx, row in data.iterrows():
        if row['gt_detection_win'] == 1 and not in_window:
            in_window = True
            start_idx = idx
        elif row['gt_detection_win'] == 0 and in_window:
            in_window = False
            windows.append((start_idx, idx))
    
    if in_window:
        windows.append((start_idx, len(data)))
    
    return windows

def normalize_pressure(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize pressure values using Z-score standardization."""
    data = data.copy()
    mean_pressure = data['PRESSURE'].mean()
    std_pressure = data['PRESSURE'].std()
    data['PRESSURE'] = (data['PRESSURE'] - mean_pressure) / std_pressure
    return data

def plot_confidence_distribution(y_true, y_pred_proba, save_path='confidence_distribution.png'):
    """Plot distribution of confidence values for each class."""
    plt.figure(figsize=(12, 6))
    
    # Get confidence values for each class
    vortex_conf = y_pred_proba[y_true == 1]
    non_vortex_conf = y_pred_proba[y_true == 0]
    
    # Plot histograms
    plt.hist(vortex_conf, bins=50, alpha=0.5, label='Vortex', color='red')
    plt.hist(non_vortex_conf, bins=50, alpha=0.5, label='Non-Vortex', color='blue')
    
    plt.title('Distribution of Confidence Values')
    plt.xlabel('Confidence Value')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confidence_timeline(data, y_pred_proba, detection_windows, save_path='confidence_timeline.png'):
    """Plot confidence values over time with vortex events marked."""
    plt.figure(figsize=(15, 6))
    
    # Create time index
    time_index = range(len(y_pred_proba))
    
    # Plot confidence values
    plt.plot(time_index, y_pred_proba, label='Confidence', color='blue', alpha=0.7)
    
    # Mark detection windows
    for start, end in detection_windows:
        plt.axvspan(start, end, color='red', alpha=0.2, label='Detection Window' if start == detection_windows[0][0] else "")
    
    plt.title('Confidence Values Over Time')
    plt.xlabel('Time Index')
    plt.ylabel('Confidence Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_on_full_dataset(model: VortexLSTMModel, data: pd.DataFrame) -> dict:
    """Evaluate model performance on the full dataset."""
    print("\nPreparing sequences from full dataset (no sampling)...")
    X_full, y_full = model.prepare_sequences(data, apply_sampling=False)
    
    print("Making predictions on full dataset...")
    y_pred_proba = model.predict(X_full)
    
    # Try different thresholds to find the best F1 score
    best_f1 = 0
    best_threshold = 0.5
    thresholds = np.linspace(0.3, 0.7, 41)
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_full, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Best F1 score: {best_f1:.4f}")
    
    # Use the best threshold for final evaluation
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    precision = precision_score(y_full, y_pred)
    recall = recall_score(y_full, y_pred)
    f1 = f1_score(y_full, y_pred)
    roc_auc = roc_auc_score(y_full, y_pred_proba)
    pr_auc = average_precision_score(y_full, y_pred_proba)
    
    # Print class distribution
    print("\nClass distribution in full dataset:")
    print(f"Vortex sequences: {sum(y_full)}")
    print(f"Non-vortex sequences: {len(y_full) - sum(y_full)}")
    print(f"Ratio: {(len(y_full) - sum(y_full)) / sum(y_full):.2f}:1")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_true': y_full,
        'threshold': best_threshold
    }

def analyze_pressure_patterns(data: pd.DataFrame, window_size: int = 60):
    """Analyze pressure patterns around vortices vs non-vortices."""
    # Get indices of vortex events
    vortex_indices = np.where(data['gt_detection_win'] == 1)[0]
    
    # Sample some non-vortex indices
    non_vortex_indices = np.random.choice(
        np.where(data['gt_detection_win'] == 0)[0],
        size=min(1000, len(vortex_indices)),
        replace=False
    )
    
    # Collect pressure patterns
    vortex_patterns = []
    non_vortex_patterns = []
    
    for idx in vortex_indices:
        if idx >= window_size and idx + window_size < len(data):
            pattern = data['PRESSURE'].iloc[idx-window_size:idx+window_size].values
            vortex_patterns.append(pattern)
    
    for idx in non_vortex_indices:
        if idx >= window_size and idx + window_size < len(data):
            pattern = data['PRESSURE'].iloc[idx-window_size:idx+window_size].values
            non_vortex_patterns.append(pattern)
    
    # Calculate statistics
    vortex_patterns = np.array(vortex_patterns)
    non_vortex_patterns = np.array(non_vortex_patterns)
    
    print("\nPressure Pattern Analysis:")
    print(f"Number of vortex patterns analyzed: {len(vortex_patterns)}")
    print(f"Number of non-vortex patterns analyzed: {len(non_vortex_patterns)}")
    
    # Calculate mean patterns
    mean_vortex = np.mean(vortex_patterns, axis=0)
    mean_non_vortex = np.mean(non_vortex_patterns, axis=0)
    
    # Calculate standard deviations
    std_vortex = np.std(vortex_patterns, axis=0)
    std_non_vortex = np.std(non_vortex_patterns, axis=0)
    
    # Plot the patterns
    plt.figure(figsize=(12, 6))
    
    # Plot mean patterns
    plt.plot(mean_vortex, label='Vortex', color='red')
    plt.plot(mean_non_vortex, label='Non-Vortex', color='blue')
    
    # Plot standard deviation ranges
    plt.fill_between(range(len(mean_vortex)), 
                    mean_vortex - std_vortex, 
                    mean_vortex + std_vortex, 
                    color='red', alpha=0.2)
    plt.fill_between(range(len(mean_non_vortex)), 
                    mean_non_vortex - std_non_vortex, 
                    mean_non_vortex + std_non_vortex, 
                    color='blue', alpha=0.2)
    
    plt.title('Mean Pressure Patterns Around Vortices vs Non-Vortices')
    plt.xlabel('Time Steps')
    plt.ylabel('Pressure (Pa)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / 'pressure_patterns.png')
    plt.close()
    
    # Print some statistics
    print("\nPattern Statistics:")
    print(f"Mean pressure in vortex patterns: {np.mean(mean_vortex):.2f} ± {np.mean(std_vortex):.2f}")
    print(f"Mean pressure in non-vortex patterns: {np.mean(mean_non_vortex):.2f} ± {np.mean(std_non_vortex):.2f}")
    print(f"Max pressure difference: {np.max(np.abs(mean_vortex - mean_non_vortex)):.2f}")
    print(f"Average pressure difference: {np.mean(np.abs(mean_vortex - mean_non_vortex)):.2f}")

def main():
    """Main function to train and evaluate the LSTM model."""
    parser = argparse.ArgumentParser(description='Train or evaluate LSTM model')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    parser.add_argument('--analyze', action='store_true', help='Analyze pressure patterns')
    parser.add_argument('--full_eval', action='store_true', help='Run evaluation on full dataset')
    args = parser.parse_args()
    
    print("Starting LSTM model training...")
    
    # Load data
    print("Loading data...")
    start_time = time.time()
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Print pressure statistics
    print("\nPressure Statistics:")
    print(f"Mean: {data['PRESSURE'].mean():.2f}")
    print(f"Std: {data['PRESSURE'].std():.2f}")
    print(f"Min: {data['PRESSURE'].min():.2f}")
    print(f"Max: {data['PRESSURE'].max():.2f}")
    
    if args.analyze:
        print("\nAnalyzing pressure patterns...")
        analyze_pressure_patterns(data)
        return
    
    # Split data temporally (70/15/15)
    n_samples = len(data)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    print("\nSplit statistics:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize model
    model = VortexLSTMModel(window_size=60)
    
    # Prepare sequences for each split
    print("\nPreparing training sequences...")
    X_train, y_train = model.prepare_sequences(train_data, apply_sampling=True)
    
    print("\nPreparing validation sequences...")
    X_val, y_val = model.prepare_sequences(val_data, apply_sampling=True)
    
    print("\nPreparing test sequences...")
    X_test, y_test = model.prepare_sequences(test_data, apply_sampling=False)
    
    # Print class distribution
    print("\nClass distribution in sets:")
    print(f"Training - Vortex: {sum(y_train)}, Non-vortex: {len(y_train) - sum(y_train)}")
    print(f"Validation - Vortex: {sum(y_val)}, Non-vortex: {len(y_val) - sum(y_val)}")
    print(f"Test - Vortex: {sum(y_test)}, Non-vortex: {len(y_test) - sum(y_test)}")
    
    # Model path
    model_path = Path(__file__).parent.parent / 'models' / 'lstm_model.h5'
    
    # Train or load model
    if model_path.exists() and not args.retrain:
        print("\nLoading existing model...")
        model.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': model.focal_loss()}
        )
        print("Model loaded successfully")
    else:
        if args.retrain:
            print("\nForce retrain flag set. Training new model...")
        else:
            print("\nNo model found. Training new model...")
        
        # Train model
        print("\nTraining LSTM model...")
        history = model.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=128)
        
        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.model.save(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_results = model.evaluate(X_test, y_test)
    
    # Print test results
    print("\nTest Set Performance:")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1']:.4f}")
    print(f"ROC-AUC: {test_results['roc_auc']:.4f}")
    print(f"PR-AUC: {test_results['pr_auc']:.4f}")
    
    # Generate test set visualizations
    print("\nGenerating test set visualizations...")
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_lstm_metrics(
        model=model.model,
        X_test=X_test,
        y_test=y_test,
        y_pred=test_results['y_pred'],
        y_pred_proba=test_results['y_pred_proba'],
        model_name='LSTM Model (Test Set)',
        save_dir=results_dir
    )
    
    # Full dataset evaluation (optional)
    if args.full_eval:
        print("\nEvaluating model on full dataset...")
        full_results = evaluate_on_full_dataset(model, data)
        
        print("\nFull Dataset Performance:")
        print(f"Precision: {full_results['precision']:.4f}")
        print(f"Recall: {full_results['recall']:.4f}")
        print(f"F1-Score: {full_results['f1']:.4f}")
        print(f"ROC-AUC: {full_results['roc_auc']:.4f}")
        print(f"PR-AUC: {full_results['pr_auc']:.4f}")
        
        # Generate full dataset visualizations
        print("\nGenerating full dataset visualizations...")
        visualize_lstm_metrics(
            model=model.model,
            X_test=X_full,
            y_test=full_results['y_true'],
            y_pred=full_results['y_pred'],
            y_pred_proba=full_results['y_pred_proba'],
            model_name='LSTM Model (Full Dataset)',
            save_dir=results_dir
        )
    
    create_lstm_report('LSTM Model', results_dir)
    
    # Plot training history
    if args.retrain and 'history' in locals():
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print("\nModel training and analysis complete!")

if __name__ == "__main__":
    main() 
