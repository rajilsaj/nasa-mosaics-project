#!/usr/bin/env python3
"""
Two-Stage Pipeline for Vortex Detection

This pipeline orchestrates the autoencoder and classifier models
without coupling them together. Each model can be used independently.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Import the standalone models
from lstm_autoencoder import VortexLSTMAutoencoder
from classifier_model import VortexClassifier

def debug_print(debug: bool, *args, **kwargs):
    """Print debug information if debug flag is set."""
    if debug:
        print(*args, **kwargs)

class TwoStagePipeline:
    """Pipeline that orchestrates autoencoder and classifier without coupling."""
    
    def __init__(self, window_size=60, debug=False):
        """Initialize the pipeline."""
        self.window_size = window_size
        self.debug = debug
        self.autoencoder = VortexLSTMAutoencoder(window_size=window_size, debug=debug)
        self.classifier = VortexClassifier(window_size=window_size, debug=debug)
        
    def debug_print(self, *args, **kwargs):
        """Print debug information if debug flag is set."""
        debug_print(self.debug, *args, **kwargs)
    
    def prepare_sequences(self, data, apply_detrending=True):
        """
        Prepare sequences from data for both stages.
        
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
        
        self.debug_print(f"\nPreparing sequences...")
        self.debug_print(f"Total data points: {len(data)}")
        
        # Process all data with sliding window
        for i in range(self.window_size, len(data)):
            # Get the pressure window
            pressure_window = pressure_values[i-self.window_size:i].copy()
            
            if apply_detrending:
                # Apply local detrending (subtract window mean)
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
        
        self.debug_print(f"Prepared {len(sequences)} sequences")
        self.debug_print(f"Vortex windows: {sum(labels)}")
        self.debug_print(f"Normal windows: {len(labels) - sum(labels)}")
        
        return sequences, labels
    
    def train_autoencoder(self, data, epochs=50, batch_size=128, learning_rate=0.001):
        """Train the autoencoder stage."""
        print("\n=== Stage 1: Training Autoencoder ===")
        
        # Prepare normal sequences for autoencoder
        normal_sequences = self.autoencoder.prepare_normal_sequences_for_autoencoder(data)
        
        if len(normal_sequences) > 0:
            print(f"Training autoencoder on {len(normal_sequences)} normal sequences...")
            
            # Train autoencoder
            autoencoder_history = self.autoencoder.train(
                normal_sequences,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Set threshold
            self.autoencoder.set_autoencoder_threshold(normal_sequences)
            
            print("Autoencoder training completed!")
            return autoencoder_history
        else:
            print("Warning: No normal sequences found for autoencoder training")
            return None
    
    def train_classifier(self, data, epochs=50, batch_size=256, learning_rate=0.001):
        """Train the classifier stage."""
        print("\n=== Stage 2: Training Classifier ===")
        
        # Prepare sequences for classifier
        sequences, labels = self.prepare_sequences(data)
        
        # Split data sequentially
        split_idx = int(0.8 * len(sequences))
        X_train_full = sequences[:split_idx]
        y_train_full = labels[:split_idx]
        X_test = sequences[split_idx:]
        y_test = labels[split_idx:]
        
        # Further split training data
        val_split_idx = int(0.8 * len(X_train_full))
        X_train = X_train_full[:val_split_idx]
        y_train = y_train_full[:val_split_idx]
        X_val = X_train_full[val_split_idx:]
        y_val = y_train_full[val_split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Filter training data with autoencoder
        print("Filtering training data with autoencoder...")
        filtered_X_train, filtered_indices = self.autoencoder.filter_with_autoencoder(X_train)
        filtered_y_train = y_train[filtered_indices]
        
        # Filter validation data with autoencoder
        filtered_X_val, filtered_val_indices = self.autoencoder.filter_with_autoencoder(X_val)
        filtered_y_val = y_val[filtered_val_indices]
        
        print(f"Filtered training data: {len(filtered_X_train)} sequences")
        print(f"Filtered validation data: {len(filtered_X_val)} sequences")
        
        if len(filtered_X_train) > 0 and len(filtered_X_val) > 0:
            # Train classifier
            classifier_history = self.classifier.train(
                filtered_X_train, filtered_y_train,
                filtered_X_val, filtered_y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            print("Classifier training completed!")
            return classifier_history
        else:
            print("Warning: No sequences passed autoencoder filter for training")
            return None
    
    def predict(self, X):
        """Two-stage prediction: autoencoder filter + classifier."""
        self.debug_print(f"\nRunning two-stage prediction...")
        self.debug_print(f"Input sequences: {len(X)}")
        
        # Stage 1: Autoencoder filtering
        filtered_X, filtered_indices = self.autoencoder.filter_with_autoencoder(X)
        
        if len(filtered_X) == 0:
            self.debug_print("No sequences passed autoencoder filter")
            return np.zeros(len(X)), np.zeros(len(X))
        
        # Stage 2: Classifier prediction
        filtered_predictions = self.classifier.predict(filtered_X)
        
        # Create full prediction array
        full_predictions = np.zeros(len(X))
        full_predictions[filtered_indices] = filtered_predictions.flatten()
        
        self.debug_print(f"Final predictions:")
        self.debug_print(f"  Positive predictions: {np.sum(full_predictions > 0.5)}")
        self.debug_print(f"  Mean prediction: {np.mean(full_predictions):.4f}")
        
        return full_predictions, filtered_indices
    
    def evaluate(self, X_test, y_test):
        """Evaluate the two-stage pipeline."""
        self.debug_print(f"\nEvaluating two-stage pipeline...")
        
        # Get predictions
        y_pred_proba, filtered_indices = self.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'average_precision': ap,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'filtered_ratio': len(filtered_indices) / len(X_test)
        }
        
        print(f"\nTwo-Stage Pipeline Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  Filtered ratio: {results['filtered_ratio']:.3f}")
        
        return results
    
    def save_models(self, autoencoder_path='autoencoder_model.h5', classifier_path='classifier_model.h5'):
        """Save both models."""
        self.autoencoder.autoencoder.save(autoencoder_path)
        self.debug_print(f"Autoencoder saved to: {autoencoder_path}")
        
        self.classifier.save_model(classifier_path)
        self.debug_print(f"Classifier saved to: {classifier_path}")
    
    def load_models(self, autoencoder_path='autoencoder_model.h5', classifier_path='classifier_model.h5'):
        """Load both models."""
        if Path(autoencoder_path).exists():
            self.autoencoder.autoencoder = tf.keras.models.load_model(autoencoder_path)
            self.debug_print(f"Autoencoder loaded from: {autoencoder_path}")
        
        if Path(classifier_path).exists():
            self.classifier.load_model(classifier_path)
            self.debug_print(f"Classifier loaded from: {classifier_path}")

def main():
    """Main function to train and evaluate the two-stage pipeline."""
    parser = argparse.ArgumentParser(description='Two-Stage Pipeline for Vortex Detection')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--window_size', type=int, default=60, help='Window size for sequences')
    parser.add_argument('--autoencoder_epochs', type=int, default=50, help='Autoencoder training epochs')
    parser.add_argument('--classifier_epochs', type=int, default=50, help='Classifier training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the models')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TwoStagePipeline(window_size=args.window_size, debug=args.debug)
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    data = pd.read_csv(args.data_path)
    print(f"Loaded {len(data)} samples")
    
    # Prepare sequences
    sequences, labels = pipeline.prepare_sequences(data)
    
    # Split data sequentially (correct for time series)
    split_idx = int(0.8 * len(sequences))
    X_train_full = sequences[:split_idx]
    y_train_full = labels[:split_idx]
    X_test = sequences[split_idx:]
    y_test = labels[split_idx:]
    
    # Further split training data
    val_split_idx = int(0.8 * len(X_train_full))
    X_train = X_train_full[:val_split_idx]
    y_train = y_train_full[:val_split_idx]
    X_val = X_train_full[val_split_idx:]
    y_val = y_train_full[val_split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Model paths
    autoencoder_path = 'autoencoder_model.h5'
    classifier_path = 'classifier_model.h5'
    
    # Train or load models
    if Path(autoencoder_path).exists() and Path(classifier_path).exists() and not args.retrain:
        print("\nLoading existing models...")
        pipeline.load_models(autoencoder_path, classifier_path)
        print("Models loaded successfully")
    else:
        if args.retrain:
            print("\nForce retrain flag set. Training new models...")
        else:
            print("\nNo models found. Training new models...")
        
        # Stage 1: Train autoencoder
        autoencoder_history = pipeline.train_autoencoder(
            data,
            epochs=args.autoencoder_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Stage 2: Train classifier
        classifier_history = pipeline.train_classifier(
            data,
            epochs=args.classifier_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Save models
        pipeline.save_models(autoencoder_path, classifier_path)
        print(f"\nModels saved to: {autoencoder_path} and {classifier_path}")
    
    # Evaluate model on test set
    print("\nEvaluating two-stage pipeline on test set...")
    results = pipeline.evaluate(X_test, y_test)
    
    print("\nTest Set Performance:")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")
    print(f"AUC: {results['auc']:.4f}")
    print(f"Average Precision: {results['average_precision']:.4f}")
    print(f"Filtered ratio: {results['filtered_ratio']:.3f}")

if __name__ == "__main__":
    main() 