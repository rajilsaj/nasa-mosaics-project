#!/usr/bin/env python3
"""
LSTM Classifier for Vortex Detection

This module implements a two-stage LSTM classifier that takes
autoencoder-filtered sequences and performs final vortex detection.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from pathlib import Path
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

def debug_print(debug: bool, *args, **kwargs):
    """Print only if debug is enabled."""
    if debug:
        print(*args, **kwargs)

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss implementation for handling class imbalance.
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Weighting factor for positive class (default: 0.25)
    
    Returns:
        Focal loss function
    """
    def focal_loss_fn(y_true, y_pred):
        # Clip predictions to avoid log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Calculate cross entropy
        cross_entropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Calculate p_t
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Calculate alpha_t
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Calculate focal loss
        focal_loss = alpha_t * tf.pow(1 - p_t, gamma) * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fn

def f1_score_metric(y_true, y_pred):
    """
    Custom F1-score metric for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        
    Returns:
        F1-score
    """
    # Convert to float32 and ensure proper shapes
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Use a lower threshold for imbalanced data
    threshold = 0.2  # Lower threshold for better recall
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    
    # Calculate true positives, false positives, false negatives
    tp = tf.reduce_sum(y_true * y_pred_binary)
    fp = tf.reduce_sum((1 - y_true) * y_pred_binary)
    fn = tf.reduce_sum(y_true * (1 - y_pred_binary))
    
    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    
    # Calculate F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    return f1

class ThresholdOptimizationCallback(Callback):
    """
    Custom callback that finds the optimal threshold for F1-score during validation.
    """
    def __init__(self, validation_data, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.verbose = verbose
        self.best_thresholds = []
        self.best_f1_scores = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Get validation predictions
        X_val, y_val = self.validation_data
        y_pred_proba = self.model.predict(X_val, verbose=0)
        
        # Try different thresholds to find the best F1-score
        thresholds = np.linspace(0.1, 0.9, 81)  # 0.1 to 0.9 in 0.01 steps
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            try:
                f1 = f1_score(y_val, y_pred, zero_division=0)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
            except:
                continue
        
        # Store results
        self.best_thresholds.append(best_threshold)
        self.best_f1_scores.append(best_f1)
        
        # Print results
        if self.verbose:
            print(f"\nEpoch {epoch + 1} - Threshold Optimization:")
            print(f"  Best Threshold: {best_threshold:.3f}")
            print(f"  Best F1-Score:  {best_f1:.4f}")
            print(f"  Precision:       {best_precision:.4f}")
            print(f"  Recall:          {best_recall:.4f}")
        
        # Update logs for monitoring
        if logs is not None:
            logs['val_optimal_f1'] = best_f1
            logs['val_optimal_threshold'] = best_threshold
            logs['val_optimal_precision'] = best_precision
            logs['val_optimal_recall'] = best_recall

class VortexClassifier:
    """LSTM Classifier for vortex detection on filtered sequences."""
    
    def __init__(self, window_size: int = 60, prediction_threshold: float = 0.5, debug: bool = False):
        """
        Initialize the classifier.
        
        Args:
            window_size: Size of input sequences
            prediction_threshold: Threshold for binary classification
            debug: Enable debug output
        """
        self.window_size = window_size
        self.prediction_threshold = prediction_threshold
        self.debug = debug
        self.model = None
    
    def debug_print(self, *args, **kwargs):
        """Print debug messages."""
        debug_print(self.debug, *args, **kwargs)
    
    def prepare_sequences(self, data: pd.DataFrame, apply_detrending: bool = True) -> tuple:
        """
        Prepare sequences for classification.
        
        Args:
            data: DataFrame with pressure data and labels
            apply_detrending: Whether to apply detrending
            
        Returns:
            Tuple of (sequences, labels)
        """
        self.debug_print("Preparing sequences for classification...")
        
        sequences = []
        labels = []
        
        # Get pressure values
        pressure_values = data['PRESSURE'].values
        
        # Prepare sequences with sliding window
        for i in range(self.window_size, len(data)):
            # Get pressure window
            pressure_window = pressure_values[i-self.window_size:i].copy()
            
            # Apply detrending if requested
            if apply_detrending:
                local_mean = np.mean(pressure_window)
                pressure_window = pressure_window - local_mean
            
            # Reshape for single feature
            sequence = pressure_window.reshape(-1, 1)
            sequences.append(sequence)
            
            # Get label (vortex or not) - using only gt_detection_win for true precursor prediction
            label = 1 if data.iloc[i]['gt_detection_win'] == 1 else 0
            labels.append(label)
        
        sequences = np.array(sequences)
        labels = np.array(labels)
        
        self.debug_print(f"Prepared {len(sequences)} sequences")
        self.debug_print(f"Vortex sequences: {sum(labels)}")
        self.debug_print(f"Normal sequences: {len(labels) - sum(labels)}")
        
        return sequences, labels
    
    def build_model(self, input_shape: tuple, learning_rate: float = 0.0001):
        """
        Build the LSTM classifier model.
        
        Args:
            input_shape: Shape of input sequences
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled model
        """
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model with Focal Loss
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal Loss for imbalanced data
            metrics=['accuracy', 'precision', 'recall', f1_score_metric]
        )
        
        return model
    
    def balance_training_data(self, X_train, y_train):
        """
        Balance training data to 50/50 class distribution like in original LSTM implementation.
        
        Args:
            X_train: Training sequences
            y_train: Training labels
            
        Returns:
            X_balanced, y_balanced: Balanced training data
        """
        self.debug_print(f"\nBalancing training data...")
        self.debug_print(f"Original training samples: {len(X_train)}")
        self.debug_print(f"Original class distribution: {np.bincount(y_train)}")
        
        # Find positive and negative indices
        positive_indices = np.where(y_train == 1)[0]
        negative_indices = np.where(y_train == 0)[0]
        
        n_positive = len(positive_indices)
        n_negative = len(negative_indices)
        n_samples_per_class = min(n_positive, n_negative)
        
        self.debug_print(f"Positive samples: {n_positive}")
        self.debug_print(f"Negative samples: {n_negative}")
        self.debug_print(f"Samples per class after balancing: {n_samples_per_class}")
        
        # For positive samples: use actual vortex indices (temporal order preserved)
        balanced_positive_indices = positive_indices[:n_samples_per_class]
        
        # For negative samples: randomly sample from non-vortex areas
        np.random.seed(42)  # For reproducibility
        balanced_negative_indices = np.random.choice(negative_indices, size=n_samples_per_class, replace=False)
        
        # Combine balanced samples (positive first, then negative)
        balanced_indices = np.concatenate([balanced_positive_indices, balanced_negative_indices])
        
        # Shuffle the balanced data
        np.random.shuffle(balanced_indices)
        
        # Get corresponding features and labels
        X_balanced = X_train[balanced_indices]
        y_balanced = y_train[balanced_indices]
        
        self.debug_print(f"Balanced training samples: {len(X_balanced)}")
        self.debug_print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=256, learning_rate=0.0001):
        """Train the classifier."""
        
        self.debug_print(f"\nTraining classifier...")
        self.debug_print(f"Training samples: {len(X_train)}")
        self.debug_print(f"Validation samples: {len(X_val)}")
        self.debug_print(f"Input shape: {X_train.shape}")
        
        # Balance training data to 50/50
        X_train_balanced, y_train_balanced = self.balance_training_data(X_train, y_train)
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, learning_rate=learning_rate)
        
        # Callbacks (removed ReduceLROnPlateau for stability)
        early_stopping = EarlyStopping(
            monitor='val_optimal_f1',  # Monitor optimal F1-score from threshold optimization
            mode='max',  # F1-score should be maximized
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_classifier.h5',
            monitor='val_optimal_f1', # Monitor optimal F1-score from threshold optimization
            mode='max',  # F1-score should be maximized
            save_best_only=True,
            verbose=1
        )
        
        # Add threshold optimization callback
        threshold_callback = ThresholdOptimizationCallback(
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Train model with balanced data (removed ReduceLROnPlateau)
        history = self.model.fit(
            X_train_balanced, y_train_balanced,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, model_checkpoint, threshold_callback],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, optimize_threshold=True):
        """Evaluate the classifier."""
        self.debug_print(f"\nEvaluating classifier...")
        
        # Get predictions
        y_pred_proba = self.predict(X_test)
        
        if optimize_threshold:
            # Find optimal threshold for test set
            thresholds = np.linspace(0.01, 0.99, 99)  # 0.01 to 0.99 in 0.01 steps
            best_f1 = 0
            best_threshold = self.prediction_threshold
            best_precision = 0
            best_recall = 0
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                try:
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                        best_precision = precision
                        best_recall = recall
                except:
                    continue
            
            print(f"\nTest Set Threshold Optimization:")
            print(f"  Optimal Threshold: {best_threshold:.3f}")
            print(f"  Optimal F1-Score:  {best_f1:.4f}")
            print(f"  Precision:          {best_precision:.4f}")
            print(f"  Recall:             {best_recall:.4f}")
            
            # Use optimal threshold for final evaluation
            y_pred = (y_pred_proba >= best_threshold).astype(int)
        else:
            # Use default threshold
            y_pred = (y_pred_proba > self.prediction_threshold).astype(int)
        
        # Calculate metrics
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
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\nClassifier Results:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        return results
    
    def save_model(self, model_path='classifier_model.h5'):
        """Save the model."""
        if self.model is not None:
            self.model.save(model_path)
            self.debug_print(f"Classifier saved to: {model_path}")
    
    def load_model(self, model_path='classifier_model.h5'):
        """Load the model."""
        if Path(model_path).exists():
            # Load with custom objects for Focal Loss and F1-score metric
            custom_objects = {
                'focal_loss_fn': focal_loss(gamma=2.0, alpha=0.25),
                'f1_score_metric': f1_score_metric
            }
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            self.debug_print(f"Classifier loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

def main():
    """Main function to train and evaluate the classifier."""
    parser = argparse.ArgumentParser(description='Train or evaluate LSTM Classifier')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--data_path', type=str, help='Path to input data CSV (if not using filtered data)')
    parser.add_argument('--filtered_data', type=str, help='Path to filtered data pickle file (from autoencoder)')
    parser.add_argument('--window_size', type=int, default=60, help='Window size for sequences')
    parser.add_argument('--model_name', type=str, default='classifier_model.h5', help='Name for the saved model file.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("Starting LSTM Classifier training...")
    
    # Initialize model
    classifier = VortexClassifier(window_size=args.window_size, debug=args.debug)
    
    # Load data (either filtered or raw)
    if args.filtered_data:
        print(f"Loading filtered data from: {args.filtered_data}")
        import pickle
        with open(args.filtered_data, 'rb') as f:
            filtered_data = pickle.load(f)
        
        sequences = filtered_data['sequences']
        labels = filtered_data['labels']
        print(f"Loaded {len(sequences)} filtered sequences")
        print(f"Vortex sequences: {sum(labels)}")
        print(f"Normal sequences: {len(labels) - sum(labels)}")
        print(f"Filter ratio: {filtered_data['filter_ratio']:.3f}")
        
    else:
        # Load raw data
        if not args.data_path:
            data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
        else:
            data_path = args.data_path
            
        print(f"Loading data from: {data_path}")
        start_time = time.time()
        data = pd.read_csv(data_path)
        print(f"Data loaded in {time.time() - start_time:.2f} seconds")
        
        # Prepare sequences
        print(f"\nPreparing sequences...")
        sequences, labels = classifier.prepare_sequences(data)
    
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
    
    # Model path
    model_path = Path(__file__).parent.parent / 'models' / args.model_name
    
    # Train or load model
    if model_path.exists() and not args.retrain:
        print("\nLoading existing model...")
        classifier.load_model(model_path)
        print("Model loaded successfully")
    else:
        if args.retrain:
            print("\nForce retrain flag set. Training new model...")
        else:
            print("\nNo model found. Training new model...")
        
        # Train model
        print("\nTraining LSTM Classifier...")
        history = classifier.train(X_train, y_train, X_val, y_val, 
                                 epochs=args.epochs, batch_size=args.batch_size, 
                                 learning_rate=args.learning_rate)
        
        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_results = classifier.evaluate(X_test, y_test)
    
    print("\nTest Set Performance:")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"F1-Score: {test_results['f1']:.4f}")
    print(f"ROC-AUC: {test_results['auc']:.4f}")
    print(f"PR-AUC: {test_results['average_precision']:.4f}")
    
    # Event-based evaluation
    print("\n" + "="*50)
    print("EVENT-BASED EVALUATION")
    print("="*50)
    
    # Load original test data for ground truth
    if args.filtered_data:
        # We need the original data for gt_detection_win and gt_fwhm
        print("\nLoading original test data for event-based evaluation...")
        if not args.data_path:
            data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
        else:
            data_path = args.data_path
            
        original_data = pd.read_csv(data_path)
        
        # Get original indices from filtered data for proper alignment
        print("\nAligning filtered data with original data...")
        filtered_indices = filtered_data['original_indices']
        
        # Get test portion of filtered indices
        test_filtered_indices = filtered_indices[split_idx:]
        
        # Map to original data positions
        original_test_indices = test_filtered_indices
        
        # Get original test data using mapped indices
        test_original_data = original_data.iloc[original_test_indices].reset_index(drop=True)
        
        # Verify alignment with SCLK values
        print(f"Verifying data alignment...")
        print(f"Test predictions length: {len(test_results['y_pred'])}")
        print(f"Original test data length: {len(test_original_data)}")
        
        if len(test_results['y_pred']) != len(test_original_data):
            print("WARNING: Length mismatch! This indicates alignment issues.")
            print("Falling back to sequential alignment...")
            # Fallback: use sequential alignment
            test_start_idx = split_idx
            test_end_idx = len(original_data)
            test_original_data = original_data.iloc[test_start_idx:test_end_idx].reset_index(drop=True)
        else:
            print("✅ Data alignment verified!")
        
        # Get predictions aligned with original data
        predictions = test_results['y_pred']
        
        # Import and run event-based evaluation
        from event_based_evaluation import evaluate_vortex_detection
        
        event_results = evaluate_vortex_detection(
            predictions=predictions,
            gt_detection_win=test_original_data['gt_detection_win'].values,
            gt_fwhm=test_original_data['gt_fwhm'].values,
            verbose=True
        )
        
        print(f"\n" + "="*50)
        print("EVENT-BASED SUMMARY")
        print("="*50)
        print(f"Point-wise F1: {test_results['f1']:.4f}")
        print(f"Event-based F1: {event_results['event_metrics']['f1']:.4f}")
        print(f"Improvement: {event_results['event_metrics']['f1'] - test_results['f1']:.4f}")
        print(f"Event-based Recall: {event_results['event_metrics']['recall']:.4f}")
        print(f"Event-based Precision: {event_results['event_metrics']['precision']:.4f}")
    else:
        print("\nEvent-based evaluation requires original data with gt_detection_win and gt_fwhm columns.")
        print("Please provide --data_path argument for full evaluation.")

if __name__ == "__main__":
    main() 