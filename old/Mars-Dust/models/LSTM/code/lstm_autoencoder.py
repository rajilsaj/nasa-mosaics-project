"""
LSTM Autoencoder for Vortex Detection Pre-filter

This model trains on normal (non-vortex) pressure patterns and uses reconstruction error
as an anomaly score to identify potential vortex events.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt

class VortexLSTMAutoencoder:
    """LSTM Autoencoder for anomaly detection using raw pressure data."""
    
    def __init__(self, window_size=60, debug=False):
        """Initialize the autoencoder."""
        self.window_size = window_size
        self.debug = debug
        self.model = None
        self.reconstruction_threshold = None
        
    def debug_print(self, *args, **kwargs):
        """Print debug information if debug flag is set."""
        if self.debug:
            print(*args, **kwargs)
    
    def prepare_normal_sequences(self, data, apply_detrending=True):
        """
        Prepare sequences from normal (non-vortex) data for training.
        
        Args:
            data: Input data DataFrame
            apply_detrending: If True, apply local detrending to each window
        """
        pressure_values = data['PRESSURE'].values
        gt_detection = data['gt_detection_win'].values
        gt_fwhm = data['gt_fwhm'].values
        
        # Find non-vortex regions
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        normal_indices = np.where(gt_combined == 0)[0]
        
        sequences_list = []
        
        self.debug_print(f"\nPreparing normal sequences...")
        self.debug_print(f"Total data points: {len(data)}")
        self.debug_print(f"Normal (non-vortex) points: {len(normal_indices)}")
        
        # Import artifact detector
        from artifact_detector import ArtifactDetector
        artifact_detector = ArtifactDetector(window_size=self.window_size)
        
        # Extract normal sequences using sliding window
        for i in range(self.window_size, len(data)):
            # Check if this window contains any vortex events
            window_contains_vortex = np.any(gt_combined[i-self.window_size:i] == 1)
            
            if not window_contains_vortex:
                # Get the pressure window
                pressure_window = pressure_values[i-self.window_size:i].copy()
                
                # Check if this window is an artifact
                artifact_info = artifact_detector._analyze_window_for_artifacts(pressure_window, i)
                is_artifact = artifact_info['is_artifact']
                
                # Only include if it's NOT a vortex AND NOT an artifact
                if not is_artifact:
                    if apply_detrending:
                        # Apply local detrending (subtract window mean)
                        local_mean = np.mean(pressure_window)
                        pressure_window = pressure_window - local_mean
                    
                    # Reshape for LSTM: (window_size, 1)
                    sequence = pressure_window.reshape(-1, 1)
                    sequences_list.append(sequence)
        
        sequences = np.array(sequences_list)
        
        self.debug_print(f"Prepared {len(sequences)} normal sequences (after filtering artifacts)")
        self.debug_print(f"Sequence shape: {sequences.shape}")
        
        # Debug: Show filtering stats
        total_windows = len(data) - self.window_size
        vortex_windows = sum([np.any(gt_combined[i-self.window_size:i] == 1) for i in range(self.window_size, len(data))])
        artifact_windows = total_windows - len(sequences) - vortex_windows
        self.debug_print(f"Filtering statistics:")
        self.debug_print(f"  Total windows: {total_windows}")
        self.debug_print(f"  Vortex windows (excluded): {vortex_windows}")
        self.debug_print(f"  Artifact windows (excluded): {artifact_windows}")
        self.debug_print(f"  Clean normal windows (included): {len(sequences)}")
        
        # Debug: Check if we have any extreme values
        if len(sequences) > 0:
            all_values = sequences.flatten()
            self.debug_print(f"Training data statistics:")
            self.debug_print(f"  Min: {np.min(all_values):.4f}")
            self.debug_print(f"  Max: {np.max(all_values):.4f}")
            self.debug_print(f"  Mean: {np.mean(all_values):.4f}")
            self.debug_print(f"  Std: {np.std(all_values):.4f}")
            
            # Check for extreme outliers
            extreme_count = np.sum(np.abs(all_values) > 10)
            self.debug_print(f"  Values > 10: {extreme_count}")
            self.debug_print(f"  Values < -10: {np.sum(all_values < -10)}")
            
            # Check for very extreme values
            very_extreme = np.sum(np.abs(all_values) > 50)
            self.debug_print(f"  Values > 50: {very_extreme}")
            self.debug_print(f"  Values < -50: {np.sum(all_values < -50)}")
            
            # Show the worst offenders
            if very_extreme > 0:
                worst_indices = np.argsort(np.abs(all_values))[-10:]
                self.debug_print(f"  Worst 10 values: {all_values[worst_indices]}")
        
        return sequences
    
    def prepare_test_sequences(self, data, apply_detrending=True):
        """
        Prepare sequences from all data for testing/evaluation.
        
        Args:
            data: Input data DataFrame
            apply_detrending: If True, apply local detrending to each window
        """
        pressure_values = data['PRESSURE'].values
        gt_detection = data['gt_detection_win'].values
        gt_fwhm = data['gt_fwhm'].values
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        
        sequences_list = []
        labels_list = []
        
        self.debug_print(f"\nPreparing test sequences...")
        
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
        
        sequences = np.array(sequences_list)
        labels = np.array(labels_list)
        
        self.debug_print(f"Prepared {len(sequences)} test sequences")
        self.debug_print(f"Vortex windows: {sum(labels)}")
        self.debug_print(f"Normal windows: {len(labels) - sum(labels)}")
        
        return sequences, labels
    
    def build_autoencoder(self, input_shape, learning_rate=0.001):
        """Build the LSTM autoencoder model."""
        
        # Input layer
        input_layer = Input(shape=input_shape)
        
        # Encoder
        encoder = LSTM(32, activation='relu', return_sequences=True, 
                      name='encoder_lstm')(input_layer)
        encoder = Dropout(0.1)(encoder)
        
        # Decoder (symmetric)
        decoder = LSTM(32, activation='relu', return_sequences=True, 
                      name='decoder_lstm')(encoder)
        decoder = Dropout(0.2)(decoder)
        
        # Output layer
        output = TimeDistributed(Dense(1, activation='linear'))(decoder)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',  # Mean squared error for reconstruction
            metrics=['mae']  # Mean absolute error
        )
        
        return model
    
    def train(self, X_train, epochs=50, batch_size=128, validation_split=0.2, learning_rate=0.001):
        """Train the autoencoder on normal data."""
        
        self.debug_print(f"\nTraining autoencoder...")
        self.debug_print(f"Training samples: {len(X_train)}")
        self.debug_print(f"Input shape: {X_train.shape}")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_autoencoder(input_shape, learning_rate=learning_rate)
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_autoencoder.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, X_train,  # Autoencoder: input = target
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict_reconstruction_error(self, X):
        """Predict reconstruction error for input sequences."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get reconstructions
        reconstructions = self.model.predict(X)
        
        # Calculate reconstruction error (MSE per sequence)
        reconstruction_errors = np.mean((X - reconstructions) ** 2, axis=(1, 2))
        
        return reconstruction_errors
    
    def set_threshold(self, X_normal, percentile=95):
        """Set anomaly threshold based on normal data reconstruction errors."""
        normal_errors = self.predict_reconstruction_error(X_normal)
        self.reconstruction_threshold = np.percentile(normal_errors, percentile)
        
        self.debug_print(f"\nThreshold setting:")
        self.debug_print(f"Normal reconstruction errors - Mean: {np.mean(normal_errors):.6f}")
        self.debug_print(f"Normal reconstruction errors - Std: {np.std(normal_errors):.6f}")
        self.debug_print(f"Anomaly threshold ({percentile}th percentile): {self.reconstruction_threshold:.6f}")
        
        return self.reconstruction_threshold
    
    def detect_anomalies(self, X):
        """Detect anomalies based on reconstruction error threshold."""
        if self.reconstruction_threshold is None:
            raise ValueError("Threshold not set! Call set_threshold() first.")
        
        reconstruction_errors = self.predict_reconstruction_error(X)
        anomalies = reconstruction_errors > self.reconstruction_threshold
        
        return anomalies, reconstruction_errors
    
    def evaluate(self, X_test, y_test):
        """Evaluate autoencoder performance on test data."""
        
        # Get reconstruction errors
        reconstruction_errors = self.predict_reconstruction_error(X_test)
        
        # Use reconstruction error as anomaly score (higher = more anomalous)
        anomaly_scores = reconstruction_errors
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, anomaly_scores)
        pr_auc = average_precision_score(y_test, anomaly_scores)
        
        # Try different thresholds for classification metrics
        thresholds = np.percentile(anomaly_scores, np.arange(50, 100, 5))
        best_f1 = 0
        best_threshold = None
        best_metrics = None
        
        for threshold in thresholds:
            y_pred = (anomaly_scores > threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': threshold
                }
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'best_metrics': best_metrics,
            'anomaly_scores': anomaly_scores,
            'reconstruction_errors': reconstruction_errors
        }
    
    def plot_reconstruction_examples(self, X_test, y_test, n_examples=5):
        """Plot example reconstructions for normal vs anomalous patterns."""
        
        # Get reconstructions
        reconstructions = self.model.predict(X_test)
        
        # Find examples of each class
        normal_indices = np.where(y_test == 0)[0][:n_examples]
        anomalous_indices = np.where(y_test == 1)[0][:n_examples]
        
        fig, axes = plt.subplots(2, n_examples, figsize=(3*n_examples, 6))
        
        # Plot normal examples
        for i, idx in enumerate(normal_indices):
            axes[0, i].plot(X_test[idx, :, 0], 'b-', label='Original', linewidth=2)
            axes[0, i].plot(reconstructions[idx, :, 0], 'r--', label='Reconstructed', linewidth=1)
            axes[0, i].set_title(f'Normal (Error: {self.predict_reconstruction_error(X_test[idx:idx+1])[0]:.4f})')
            axes[0, i].legend()
            axes[0, i].grid(True)
        
        # Plot anomalous examples
        for i, idx in enumerate(anomalous_indices):
            axes[1, i].plot(X_test[idx, :, 0], 'b-', label='Original', linewidth=2)
            axes[1, i].plot(reconstructions[idx, :, 0], 'r--', label='Reconstructed', linewidth=1)
            axes[1, i].set_title(f'Anomalous (Error: {self.predict_reconstruction_error(X_test[idx:idx+1])[0]:.4f})')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.savefig('autoencoder_reconstructions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Reconstruction examples saved to autoencoder_reconstructions.png")

def main():
    """Main function to train and evaluate the LSTM autoencoder."""
    
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder for vortex detection')
    parser.add_argument('--retrain', action='store_true', help='Force retraining')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--window_size', type=int, default=60, help='Window size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--threshold_percentile', type=float, default=95, help='Anomaly threshold percentile')
    parser.add_argument('--data_reduction', type=float, default=1.0, help='Reduce dataset by this fraction (0.1 = 10% of data, 1.0 = full dataset)')
    
    args = parser.parse_args()
    
    print("Starting LSTM Autoencoder training...")
    
    # Load data
    print("Loading data...")
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    
    # Apply data reduction if specified
    if args.data_reduction < 1.0:
        reduction_size = int(len(data) * args.data_reduction)
        data = data.iloc[:reduction_size]
        print(f"Reduced dataset to {len(data)} samples ({args.data_reduction*100:.1f}% of original)")
    
    # Split data temporally
    n_samples = len(data)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    print(f"Data split sizes:")
    print(f"Training: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Initialize autoencoder
    autoencoder = VortexLSTMAutoencoder(window_size=args.window_size, debug=args.debug)
    
    # Prepare normal sequences for training
    print("\nPreparing normal sequences for training...")
    X_train_normal = autoencoder.prepare_normal_sequences(train_data, apply_detrending=True)
    
    # Prepare test sequences
    print("\nPreparing test sequences...")
    X_test, y_test = autoencoder.prepare_test_sequences(test_data, apply_detrending=True)
    
    # Model path
    model_path = Path(__file__).parent / 'best_autoencoder.h5'
    
    # Train or load model
    if model_path.exists() and not args.retrain:
        print("\nLoading existing autoencoder...")
        autoencoder.model = tf.keras.models.load_model(model_path)
        print("Autoencoder loaded successfully")
    else:
        print("\nTraining new autoencoder...")
        history = autoencoder.train(
            X_train_normal, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Save model
        autoencoder.model.save(model_path)
        print(f"Autoencoder saved to: {model_path}")
    
    # Set threshold using training data
    print("\nSetting anomaly threshold...")
    autoencoder.set_threshold(X_train_normal, percentile=args.threshold_percentile)
    
    # Evaluate on test set
    print("\nEvaluating autoencoder on test set...")
    results = autoencoder.evaluate(X_test, y_test)
    
    # Print results
    print("\nAutoencoder Performance:")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"PR-AUC: {results['pr_auc']:.4f}")
    
    if results['best_metrics']:
        print(f"Best F1-Score: {results['best_metrics']['f1']:.4f}")
        print(f"Best Precision: {results['best_metrics']['precision']:.4f}")
        print(f"Best Recall: {results['best_metrics']['recall']:.4f}")
        print(f"Best Threshold: {results['best_metrics']['threshold']:.6f}")
    
    # Plot reconstruction examples
    print("\nGenerating reconstruction examples...")
    autoencoder.plot_reconstruction_examples(X_test, y_test)
    
    # Compare to random baseline
    vortex_prevalence = sum(y_test) / len(y_test)
    print(f"\nRandom PR-AUC (vortex prevalence): {vortex_prevalence:.5f}")
    if vortex_prevalence > 0:
        print(f"Autoencoder is {results['pr_auc'] / vortex_prevalence:.1f}x better than random!")
    
    print("\nAutoencoder training and evaluation complete!")

if __name__ == "__main__":
    main() 