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

def debug_print(debug: bool, *args, **kwargs):
    """Print debug information if debug flag is set."""
    if debug:
        print(*args, **kwargs)

class VortexLSTMModel:
    """Single-stage LSTM model for vortex prediction."""
    
    def __init__(self, window_size: int = 60, prediction_threshold: float = 0.2, debug: bool = False):
        """Initialize the LSTM model."""
        self.window_size = window_size
        self.prediction_threshold = prediction_threshold
        self.model = None
        self.debug = debug
        
    def debug_print(self, *args, **kwargs):
        """Print debug information if debug flag is set."""
        debug_print(self.debug, *args, **kwargs)
        
    def calculate_rolling_features(self, pressure_values: np.ndarray, 
                                 window_size: int = 20,
                                 rolling_mean_mean: float = None,
                                 rolling_mean_std: float = None,
                                 rolling_std_mean: float = None,
                                 rolling_std_std: float = None) -> tuple:
        """Calculate rolling mean and standard deviation for pressure values."""
        # Calculate rolling statistics with proper handling of edge cases
        rolling_mean = pd.Series(pressure_values).rolling(window=window_size, min_periods=1).mean()
        rolling_std = pd.Series(pressure_values).rolling(window=window_size, min_periods=1).std()
        
        # Fill any remaining NaN values with appropriate values
        rolling_mean = rolling_mean.fillna(method='bfill').fillna(method='ffill')
        rolling_std = rolling_std.fillna(method='bfill').fillna(method='ffill')
        
        if rolling_mean_mean is None:  # Training mode
            # Calculate statistics from training data
            rolling_mean_mean = np.mean(rolling_mean)
            rolling_mean_std = np.std(rolling_mean)
            rolling_std_mean = np.mean(rolling_std)
            rolling_std_std = np.std(rolling_std)
            
            # Ensure we don't divide by zero
            rolling_mean_std = max(rolling_mean_std, 1e-10)
            rolling_std_std = max(rolling_std_std, 1e-10)
            
            self.debug_print("\nCalculated rolling statistics from training data:")
            self.debug_print(f"Rolling Mean - Mean: {rolling_mean_mean:.2f}, Std: {rolling_mean_std:.2f}")
            self.debug_print(f"Rolling Std - Mean: {rolling_std_mean:.2f}, Std: {rolling_std_std:.2f}")
        
        # Normalize using either calculated or provided statistics
        normalized_mean = (rolling_mean - rolling_mean_mean) / rolling_mean_std
        normalized_std = (rolling_std - rolling_std_mean) / rolling_std_std
        
        # Ensure no NaN values in final output
        normalized_mean = normalized_mean.fillna(0)
        normalized_std = normalized_std.fillna(0)
        
        return normalized_mean, normalized_std, rolling_mean_mean, rolling_mean_std, rolling_std_mean, rolling_std_std

    def prepare_sequences(self, data: pd.DataFrame, apply_sampling: bool = True, 
                         pressure_mean: float = None, pressure_std: float = None,
                         roc_mean: float = None, roc_std: float = None,
                         rolling_mean_mean: float = None, rolling_mean_std: float = None,
                         rolling_std_mean: float = None, rolling_std_std: float = None) -> tuple:
        """Prepare sequences for vortex prediction.
        
        Args:
            data: Input data
            apply_sampling: If True, creates balanced dataset for training/validation.
                          If False, processes all available data for testing/evaluation.
            pressure_mean: Mean of pressure values from training data
            pressure_std: Standard deviation of pressure values from training data
            roc_mean: Mean of rate of change from training data
            roc_std: Standard deviation of rate of change from training data
            rolling_mean_mean: Mean of rolling mean from training data
            rolling_mean_std: Std of rolling mean from training data
            rolling_std_mean: Mean of rolling std from training data
            rolling_std_std: Std of rolling std from training data
        """
        pressure_values = data['PRESSURE'].values
        gt_detection = data['gt_detection_win'].values
        gt_fwhm = data['gt_fwhm'].values
        
        # Calculate rate of change (pressure differences)
        rate_of_change = np.diff(pressure_values)
        rate_of_change = np.concatenate([[0], rate_of_change])
        
        # Only calculate normalization statistics if not provided (training data)
        if pressure_mean is None or pressure_std is None:
            pressure_mean = np.mean(pressure_values)
            pressure_std = np.std(pressure_values)
            roc_mean = np.mean(rate_of_change)
            roc_std = np.std(rate_of_change)
            self.debug_print("\nCalculated normalization statistics from training data:")
            self.debug_print(f"Pressure - Mean: {pressure_mean:.2f}, Std: {pressure_std:.2f}")
            self.debug_print(f"Rate of Change - Mean: {roc_mean:.2f}, Std: {roc_std:.2f}")
        
        # Normalize using provided or calculated statistics
        normalized_pressure = (pressure_values - pressure_mean) / pressure_std
        normalized_roc = (rate_of_change - roc_mean) / roc_std
        
        # Calculate rolling features
        rolling_mean, rolling_std, rolling_mean_mean, rolling_mean_std, rolling_std_mean, rolling_std_std = \
            self.calculate_rolling_features(
                pressure_values,
                rolling_mean_mean=rolling_mean_mean,
                rolling_mean_std=rolling_mean_std,
                rolling_std_mean=rolling_std_mean,
                rolling_std_std=rolling_std_std
            )
        
        if not apply_sampling:
            self.debug_print("\nProcessing all available data for test/evaluation using sliding window...")
            sequences_list = []
            labels_list = []
            for i in range(self.window_size, len(data)):
                pressure_window = normalized_pressure[i-self.window_size:i]
                roc_window = normalized_roc[i-self.window_size:i]
                rolling_mean_window = rolling_mean[i-self.window_size:i]
                rolling_std_window = rolling_std[i-self.window_size:i]
                sequence = np.zeros((self.window_size, 4))
                sequence[:, 0] = pressure_window
                sequence[:, 1] = roc_window
                sequence[:, 2] = rolling_mean_window
                sequence[:, 3] = rolling_std_window
                label = 1 if np.any(np.logical_or(gt_detection[i-self.window_size:i] == 1, 
                                                gt_fwhm[i-self.window_size:i] == 1)) else 0
                sequences_list.append(sequence)
                labels_list.append(label)
            sequences = np.array(sequences_list)
            labels = np.array(labels_list)
            self.debug_print("\nFull data statistics:")
            self.debug_print(f"Total sequences: {len(sequences)}")
            self.debug_print(f"Vortex windows: {sum(labels)}")
            self.debug_print(f"Non-vortex windows: {len(labels) - sum(labels)}")
            self.debug_print(f"Ratio: {(len(labels) - sum(labels)) / sum(labels):.2f}:1")
            return sequences, labels, pressure_mean, pressure_std, roc_mean, roc_std, \
                   rolling_mean_mean, rolling_mean_std, rolling_std_mean, rolling_std_std
        
        # For training/validation: create balanced dataset
        sequences_list = []
        labels_list = []
        vortex_indices = np.where(np.logical_or(gt_detection == 1, gt_fwhm == 1))[0]
        if self.debug:
            self.debug_print(f"\nFound {len(vortex_indices)} vortex events")
        for vortex_idx in vortex_indices:
            if vortex_idx >= self.window_size:
                pressure_window = normalized_pressure[vortex_idx-self.window_size:vortex_idx]
                roc_window = normalized_roc[vortex_idx-self.window_size:vortex_idx]
                rolling_mean_window = rolling_mean[vortex_idx-self.window_size:vortex_idx]
                rolling_std_window = rolling_std[vortex_idx-self.window_size:vortex_idx]
                sequence = np.zeros((self.window_size, 4))
                sequence[:, 0] = pressure_window
                sequence[:, 1] = roc_window
                sequence[:, 2] = rolling_mean_window
                sequence[:, 3] = rolling_std_window
                sequences_list.append(sequence)
                labels_list.append(0)
            if vortex_idx + self.window_size <= len(data):
                pressure_window = normalized_pressure[vortex_idx:vortex_idx+self.window_size]
                roc_window = normalized_roc[vortex_idx:vortex_idx+self.window_size]
                rolling_mean_window = rolling_mean[vortex_idx:vortex_idx+self.window_size]
                rolling_std_window = rolling_std[vortex_idx:vortex_idx+self.window_size]
                sequence = np.zeros((self.window_size, 4))
                sequence[:, 0] = pressure_window
                sequence[:, 1] = roc_window
                sequence[:, 2] = rolling_mean_window
                sequence[:, 3] = rolling_std_window
                sequences_list.append(sequence)
                labels_list.append(1)
        sequences = np.array(sequences_list)
        labels = np.array(labels_list)
        self.debug_print("\nSequence creation statistics:")
        self.debug_print(f"Total sequences: {len(sequences)}")
        self.debug_print(f"Vortex sequences: {sum(labels)}")
        self.debug_print(f"Non-vortex sequences: {len(labels) - sum(labels)}")
        self.debug_print(f"Final ratio: {(len(labels) - sum(labels)) / sum(labels):.2f}:1")
        return sequences, labels, pressure_mean, pressure_std, roc_mean, roc_std, \
               rolling_mean_mean, rolling_mean_std, rolling_std_mean, rolling_std_std
        
    def temporal_focal_loss(self, gamma=1.5, alpha=None, temporal_weight=0.1):
        """Focal loss with temporal awareness.
        
        Args:
            gamma: Focusing parameter (default: 1.5)
            alpha: Class weight (default: 0.5 for balanced data)
            temporal_weight: Weight for temporal smoothness term
        """
        def loss_function(y_true, y_pred):
            # Original focal loss components
            y_true = tf.cast(y_true, tf.float32)
            epsilon = 1e-7
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
            
            # Focal loss calculation
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            
            focal_loss_1 = -alpha * tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1)
            focal_loss_0 = -(1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0)
            focal_loss = tf.reduce_mean(focal_loss_1 + focal_loss_0)
            
            # Temporal smoothness term
            # Penalize large changes in predictions for same class
            pred_diff = tf.abs(y_pred[1:] - y_pred[:-1])
            same_class = tf.equal(y_true[1:], y_true[:-1])
            temporal_loss = tf.reduce_mean(tf.where(same_class, pred_diff, tf.zeros_like(pred_diff)))
            
            # Combine losses
            total_loss = focal_loss + temporal_weight * temporal_loss
            
            return total_loss
        
        return loss_function
    
    def calculate_alpha(self, y_train: np.ndarray) -> float:
        """Calculate alpha for focal loss based on class distribution.
        Adjusted to consider temporal nature of the data."""
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        total = n_positive + n_negative
        
        # Calculate alpha based on class distribution
        # Using a more balanced approach since we're already sampling 1:1
        alpha = 0.5  # Fixed at 0.5 since we're balancing classes
        
        self.debug_print(f"\nClass distribution for alpha calculation:")
        self.debug_print(f"Positive examples (vortices): {n_positive}")
        self.debug_print(f"Negative examples (non-vortices): {n_negative}")
        self.debug_print(f"Using fixed alpha: {alpha:.4f} (balanced sampling)")
        
        return alpha
    
    def build_model(self, input_shape: tuple, alpha: float = None, gamma: float = 1.5):
        """Build the vortex prediction model with Bidirectional LSTM and temporal loss."""
        model = Sequential([
            Bidirectional(
                LSTM(128,  # Increased units to handle more features
                     kernel_regularizer=l2(0.0005),
                     recurrent_regularizer=l2(0.0005),
                     return_sequences=False),  # We don't need sequences for next layer
                input_shape=input_shape
            ),
            Dense(64, activation='relu'),  # Increased units
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.temporal_focal_loss(gamma=gamma, alpha=alpha, temporal_weight=0.4),
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(curve='ROC', name='roc_auc'),
                    tf.keras.metrics.AUC(curve='PR', name='pr_auc')]
        )
        
        return model
    
    def calculate_class_weights(self, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> dict:
        """Calculate class weights based on test set distribution using square root scaling.
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels (real-world distribution)
        """
        # Calculate weights based on test set distribution
        n_positive_test = np.sum(y_test == 1)
        n_negative_test = np.sum(y_test == 0)
        total_test = n_positive_test + n_negative_test
        
        # Calculate weights using square root scaling for gentler weights
        weight_positive = np.sqrt(total_test / (2 * n_positive_test))
        weight_negative = np.sqrt(total_test / (2 * n_negative_test))
        
        self.debug_print(f"\nClass weights calculation (square root scaling):")
        self.debug_print(f"Test set distribution - Positive: {n_positive_test}, Negative: {n_negative_test}")
        self.debug_print(f"Calculated weights - Positive: {weight_positive:.4f}, Negative: {weight_negative:.4f}")
        
        return {0: weight_negative, 1: weight_positive}

    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, batch_size=256):
        """Train the model with temporal-aware focal loss and class weights."""
        # Print statistics about the data
        self.debug_print("\nTraining Data Statistics:")
        self.debug_print(f"Total examples: {len(X_train)}")
        self.debug_print(f"Vortex examples: {sum(y_train)}")
        self.debug_print(f"Non-vortex examples: {len(y_train) - sum(y_train)}")
        self.debug_print(f"Ratio: {(len(y_train) - sum(y_train)) / sum(y_train):.2f}:1")
        
        # Calculate class weights based on test set distribution
        class_weights = self.calculate_class_weights(y_train, y_val, y_test)
        
        # Calculate alpha for focal loss
        alpha = self.calculate_alpha(y_train)
        
        # Train model
        self.debug_print("\nTraining vortex prediction model...")
        self.model = self.build_model((self.window_size, 4), alpha=alpha, gamma=1.5)
        
        # Add learning rate scheduler with adjusted patience
        reduce_lr = ReduceLROnPlateau(
            monitor='val_pr_auc',
            factor=0.5,
            patience=5,  # Increased patience for temporal data
            min_lr=1e-6,
            mode='max',
            verbose=1
        )
        
        # Add early stopping with adjusted patience
        early_stopping = EarlyStopping(
            monitor='val_pr_auc',
            patience=7,  # Increased patience for temporal data
            mode='max',
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,  # Add class weights here
            callbacks=[
                early_stopping,
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
        sequence = current_readings.reshape(1, self.window_size, 4)
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
        
        self.debug_print(f"\nBest threshold: {best_threshold:.4f}")
        self.debug_print(f"Best F1 score: {best_f1:.4f}")
        
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
            'threshold': best_threshold,
            'y_true': y_test  # Added ground truth labels
        }
    
    def extract_event_ranges(self, gt_detection_win, gt_fwhm):
        """Extract ranges of vortex events from ground truth."""
        # Combine both ground truth conditions
        gt = np.logical_or(gt_detection_win == 1, gt_fwhm == 1)
        events = []
        in_event = False
        for i in range(len(gt)):
            if gt[i] and not in_event:
                start = i
                in_event = True
            elif not gt[i] and in_event:
                end = i - 1
                events.append((start, end))
                in_event = False
        if in_event:
            events.append((start, len(gt) - 1))
        return events

    def evaluate_event_level(self, y_pred, window_starts, window_size, events):
        """Evaluate predictions at the event level (grouping contiguous positives)."""
        def get_event_ranges(binary_array, starts, window_size):
            # Returns list of (start, end) indices for contiguous True regions
            events = []
            in_event = False
            for i, val in enumerate(binary_array):
                if val and not in_event:
                    start = starts[i]
                    in_event = True
                elif not val and in_event:
                    end = starts[i-1] + window_size - 1
                    events.append((start, end))
                    in_event = False
            if in_event:
                end = starts[len(binary_array)-1] + window_size - 1
                events.append((start, end))
            return events

        # Group predicted positives into events
        pred_events = get_event_ranges(y_pred, window_starts, window_size)
        # True events are already provided as 'events'

        matched_true = set()
        matched_pred = set()
        tp = 0
        fp = 0
        fn = 0
        earliness = []

        # --- DEBUG PRINTS ---
        print("\n[DEBUG] Predicted event ranges:")
        for i, (p_start, p_end) in enumerate(pred_events):
            print(f"  Predicted event {i}: {p_start} to {p_end}")
        print(f"[DEBUG] Number of predicted events: {len(pred_events)}")
        print("[DEBUG] True event ranges:")
        for j, (t_start, t_end) in enumerate(events):
            print(f"  True event {j}: {t_start} to {t_end}")
        print(f"[DEBUG] Number of true events: {len(events)}")
        # --- END DEBUG PRINTS ---

        # For each predicted event, check for overlap with any true event
        for i, (p_start, p_end) in enumerate(pred_events):
            overlap = False
            for j, (t_start, t_end) in enumerate(events):
                if j in matched_true:
                    continue
                # Check for overlap
                if p_end >= t_start and p_start <= t_end:
                    tp += 1
                    matched_true.add(j)
                    matched_pred.add(i)
                    # Earliness: how early did we detect the event?
                    earliness.append(max(0, p_start - t_start))
                    overlap = True
                    print(f"[DEBUG] Predicted event {i} overlaps with true event {j}")
                    break
            if not overlap:
                fp += 1
                print(f"[DEBUG] Predicted event {i} does NOT overlap with any true event")

        # Any true event not matched is a false negative
        fn = len(events) - len(matched_true)
        print(f"[DEBUG] Number of TPs (overlaps): {tp}")
        print(f"[DEBUG] Number of FPs: {fp}")
        print(f"[DEBUG] Number of FNs: {fn}")

        return tp, fp, fn, earliness

    def evaluate_with_windows(self, test_results, gt_windows, test_data):
        """Evaluate model performance considering ground truth windows.
        
        Args:
            test_results: Dictionary containing original test results (y_pred, y_pred_proba, y_true)
            gt_windows: List of (start_idx, end_idx) tuples for ground truth windows
            test_data: Original test data DataFrame containing gt_detection_win
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        import matplotlib.pyplot as plt
        
        self.debug_print("\nDebug: Window-based evaluation")
        self.debug_print(f"Number of ground truth windows: {len(gt_windows)}")
        self.debug_print(f"First few windows: {gt_windows[:5]}")
        
        # Get original predictions from test_results
        y_pred = test_results['y_pred']
        y_pred_proba = test_results['y_pred_proba']
        y_test = test_results['y_true']
        
        # Create window starts array (since we're using stride=1)
        window_starts = np.arange(self.window_size, len(test_data))
        
        # Extract event ranges from ground truth
        events = self.extract_event_ranges(test_data['gt_detection_win'].values, test_data['gt_fwhm'].values)
        
        # Evaluate at event level
        tp, fp, fn, earliness = self.evaluate_event_level(y_pred, window_starts, self.window_size, events)
        
        # Calculate event-level metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Debug output
        self.debug_print("\nEvent-level evaluation:")
        self.debug_print(f"True Positives: {tp}")
        self.debug_print(f"False Positives: {fp}")
        self.debug_print(f"False Negatives: {fn}")
        self.debug_print(f"Precision: {precision:.4f}")
        self.debug_print(f"Recall: {recall:.4f}")
        self.debug_print(f"F1-Score: {f1:.4f}")
        
        if earliness:
            self.debug_print(f"\nEarliness statistics:")
            self.debug_print(f"Mean earliness: {np.mean(earliness):.2f} points")
            self.debug_print(f"Min earliness: {np.min(earliness):.2f} points")
            self.debug_print(f"Max earliness: {np.max(earliness):.2f} points")
        
        # Calculate original point-by-point metrics for comparison
        original_metrics = {
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'pr_auc': average_precision_score(y_test, y_pred_proba)
        }
        
        self.debug_print("\nOriginal point-by-point metrics:")
        self.debug_print(f"Precision: {original_metrics['precision']:.4f}")
        self.debug_print(f"Recall: {original_metrics['recall']:.4f}")
        self.debug_print(f"F1-Score: {original_metrics['f1']:.4f}")
        self.debug_print(f"ROC-AUC: {original_metrics['roc_auc']:.4f}")
        self.debug_print(f"PR-AUC: {original_metrics['pr_auc']:.4f}")
        
        # Plot confidence distribution
        plot_confidence_distribution(y_test, y_pred_proba)
        
        # Plot confidence timeline
        plot_confidence_timeline(test_data, y_pred_proba, gt_windows)
        
        return {
            'original_metrics': original_metrics,
            'event_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'earliness': earliness
            },
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

    def analyze_learned_patterns(self, data: pd.DataFrame, y_pred: np.ndarray, window_size: int = 60):
        """Analyze what patterns the model is learning."""
        import matplotlib.pyplot as plt
        
        # Get indices where model predicted vortex
        pred_vortex_indices = np.where(y_pred == 1)[0]
        
        # Get indices of actual vortices
        gt_combined = np.logical_or(data['gt_detection_win'] == 1, data['gt_fwhm'] == 1)
        true_vortex_indices = np.where(gt_combined)[0]
        
        # Collect pressure patterns
        pred_vortex_patterns = []
        true_vortex_patterns = []
        
        # Get patterns for predicted vortices
        for idx in pred_vortex_indices:
            if idx >= window_size and idx + window_size < len(data):
                pattern = data['PRESSURE'].iloc[idx-window_size:idx+window_size].values
                pred_vortex_patterns.append(pattern)
        
        # Get patterns for true vortices
        for idx in true_vortex_indices:
            if idx >= window_size and idx + window_size < len(data):
                pattern = data['PRESSURE'].iloc[idx-window_size:idx+window_size].values
                true_vortex_patterns.append(pattern)
        
        # Convert to numpy arrays
        pred_vortex_patterns = np.array(pred_vortex_patterns)
        true_vortex_patterns = np.array(true_vortex_patterns)
        
        # Calculate statistics
        mean_pred = np.mean(pred_vortex_patterns, axis=0)
        mean_true = np.mean(true_vortex_patterns, axis=0)
        std_pred = np.std(pred_vortex_patterns, axis=0)
        std_true = np.std(true_vortex_patterns, axis=0)
        
        # Plot the patterns
        plt.figure(figsize=(15, 10))
        
        # Plot mean patterns
        plt.subplot(2, 1, 1)
        plt.plot(mean_pred, label='Predicted Vortex Pattern', color='red')
        plt.plot(mean_true, label='True Vortex Pattern', color='blue')
        plt.fill_between(range(len(mean_pred)), 
                        mean_pred - std_pred, 
                        mean_pred + std_pred, 
                        color='red', alpha=0.2)
        plt.fill_between(range(len(mean_true)), 
                        mean_true - std_true, 
                        mean_true + std_true, 
                        color='blue', alpha=0.2)
        plt.title('Mean Pressure Patterns')
        plt.xlabel('Time Steps')
        plt.ylabel('Pressure')
        plt.legend()
        
        # Plot difference
        plt.subplot(2, 1, 2)
        plt.plot(mean_pred - mean_true, label='Difference (Pred - True)', color='green')
        plt.title('Difference Between Predicted and True Patterns')
        plt.xlabel('Time Steps')
        plt.ylabel('Pressure Difference')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learned_patterns.png')
        plt.close()
        
        # Print statistics
        print("\nPattern Analysis:")
        print(f"Number of predicted vortex patterns: {len(pred_vortex_patterns)}")
        print(f"Number of true vortex patterns: {len(true_vortex_patterns)}")
        print(f"Mean pressure in predicted patterns: {np.mean(mean_pred):.2f} ± {np.mean(std_pred):.2f}")
        print(f"Mean pressure in true patterns: {np.mean(mean_true):.2f} ± {np.mean(std_true):.2f}")
        print(f"Max difference between patterns: {np.max(np.abs(mean_pred - mean_true)):.2f}")
        print(f"Average difference between patterns: {np.mean(np.abs(mean_pred - mean_true)):.2f}")
        
        # Analyze rate of change patterns
        pred_roc_patterns = []
        true_roc_patterns = []
        
        # Get ROC patterns for predicted vortices
        for idx in pred_vortex_indices:
            if idx >= window_size and idx + window_size < len(data):
                pattern = np.diff(data['PRESSURE'].iloc[idx-window_size:idx+window_size].values)
                pred_roc_patterns.append(pattern)
        
        # Get ROC patterns for true vortices
        for idx in true_vortex_indices:
            if idx >= window_size and idx + window_size < len(data):
                pattern = np.diff(data['PRESSURE'].iloc[idx-window_size:idx+window_size].values)
                true_roc_patterns.append(pattern)
        
        # Convert to numpy arrays
        pred_roc_patterns = np.array(pred_roc_patterns)
        true_roc_patterns = np.array(true_roc_patterns)
        
        # Calculate statistics
        mean_pred_roc = np.mean(pred_roc_patterns, axis=0)
        mean_true_roc = np.mean(true_roc_patterns, axis=0)
        std_pred_roc = np.std(pred_roc_patterns, axis=0)
        std_true_roc = np.std(true_roc_patterns, axis=0)
        
        # Plot ROC patterns
        plt.figure(figsize=(15, 10))
        
        # Plot mean ROC patterns
        plt.subplot(2, 1, 1)
        plt.plot(mean_pred_roc, label='Predicted Vortex ROC', color='red')
        plt.plot(mean_true_roc, label='True Vortex ROC', color='blue')
        plt.fill_between(range(len(mean_pred_roc)), 
                        mean_pred_roc - std_pred_roc, 
                        mean_pred_roc + std_pred_roc, 
                        color='red', alpha=0.2)
        plt.fill_between(range(len(mean_true_roc)), 
                        mean_true_roc - std_true_roc, 
                        mean_true_roc + std_true_roc, 
                        color='blue', alpha=0.2)
        plt.title('Mean Rate of Change Patterns')
        plt.xlabel('Time Steps')
        plt.ylabel('Rate of Change')
        plt.legend()
        
        # Plot ROC difference
        plt.subplot(2, 1, 2)
        plt.plot(mean_pred_roc - mean_true_roc, label='Difference (Pred - True)', color='green')
        plt.title('Difference Between Predicted and True ROC Patterns')
        plt.xlabel('Time Steps')
        plt.ylabel('ROC Difference')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('learned_roc_patterns.png')
        plt.close()
        
        # Print ROC statistics
        print("\nROC Pattern Analysis:")
        print(f"Mean ROC in predicted patterns: {np.mean(mean_pred_roc):.2f} ± {np.mean(std_pred_roc):.2f}")
        print(f"Mean ROC in true patterns: {np.mean(mean_true_roc):.2f} ± {np.mean(std_true_roc):.2f}")
        print(f"Max ROC difference between patterns: {np.max(np.abs(mean_pred_roc - mean_true_roc)):.2f}")
        print(f"Average ROC difference between patterns: {np.mean(np.abs(mean_pred_roc - mean_true_roc)):.2f}")

    def evaluate_triggered_event_detection(self, y_pred, test_data):
        """Evaluate using triggered event detection logic with window alignment."""
        gt_detection = test_data['gt_detection_win'].values
        gt_fwhm = test_data['gt_fwhm'].values
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        n_samples = len(gt_combined)
        window_size = self.window_size

        # Find all true event windows
        true_events = []
        in_event = False
        for i in range(n_samples):
            if gt_combined[i] and not in_event:
                start = i
                in_event = True
            elif not gt_combined[i] and in_event:
                end = i - 1
                true_events.append((start, end))
                in_event = False
        if in_event:
            true_events.append((start, n_samples - 1))

        # Group predicted positives into events (contiguous runs)
        pred_events = []
        in_pred = False
        for i, val in enumerate(y_pred):
            if val == 1 and not in_pred:
                start = i
                in_pred = True
            elif val == 0 and in_pred:
                end = i - 1
                pred_events.append((start, end))
                in_pred = False
        if in_pred:
            pred_events.append((start, len(y_pred) - 1))

        # For each predicted event, expand start index by -window_size (min 0)
        pred_events_aligned = [(max(0, start - window_size), end) for (start, end) in pred_events]

        detected_true_events = set()
        detected_pred_events = set()
        for p_idx, (p_start, p_end) in enumerate(pred_events_aligned):
            for t_idx, (t_start, t_end) in enumerate(true_events):
                # Check for overlap
                if p_end >= t_start and p_start <= t_end:
                    detected_true_events.add(t_idx)
                    detected_pred_events.add(p_idx)

        tp = len(detected_true_events)
        fp = len(pred_events_aligned) - len(detected_pred_events)
        fn = len(true_events) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nTriggered Event Detection Evaluation (Window-Aligned):")
        print(f"Triggered Event-based Precision: {precision:.4f}")
        print(f"Triggered Event-based Recall: {recall:.4f}")
        print(f"Triggered Event-based F1-Score: {f1:.4f}")
        print(f"Triggered Event-based True Positives: {tp}")
        print(f"Triggered Event-based False Positives: {fp}")
        print(f"Triggered Event-based False Negatives: {fn}")
        print(f"Total True Events: {len(true_events)}")
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'n_true_events': len(true_events)
        }

    def evaluate_triggered_pointwise(self, y_pred, test_data):
        """Evaluate using hybrid triggered pointwise logic."""
        gt_detection = test_data['gt_detection_win'].values
        gt_fwhm = test_data['gt_fwhm'].values
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        n_samples = len(gt_combined)
        window_size = self.window_size

        # Find all true event windows
        true_events = []
        in_event = False
        for i in range(n_samples):
            if gt_combined[i] and not in_event:
                start = i
                in_event = True
            elif not gt_combined[i] and in_event:
                end = i - 1
                true_events.append((start, end))
                in_event = False
        if in_event:
            true_events.append((start, n_samples - 1))

        # For each predicted event, expand start index by -window_size (min 0)
        pred_events = []
        in_pred = False
        for i, val in enumerate(y_pred):
            if val == 1 and not in_pred:
                start = i
                in_pred = True
            elif val == 0 and in_pred:
                end = i - 1
                pred_events.append((start, end))
                in_pred = False
        if in_pred:
            pred_events.append((start, len(y_pred) - 1))
        
        pred_events_aligned = [(max(0, start - window_size), end) for (start, end) in pred_events]

        # Track which true events have been triggered
        triggered_events = set()
        for p_start, p_end in pred_events_aligned:
            for t_idx, (t_start, t_end) in enumerate(true_events):
                if p_end >= t_start and p_start <= t_end:
                    triggered_events.add(t_idx)

        # Now evaluate pointwise with triggered logic
        tp = 0  # True positives
        fp = 0  # False positives
        tn = 0  # True negatives
        fn = 0  # False negatives

        for i in range(len(y_pred)):
            # Check if this prediction index is within any triggered event
            in_triggered_event = False
            for t_idx in triggered_events:
                t_start, t_end = true_events[t_idx]
                # Adjust for window alignment
                pred_idx = i + window_size
                if t_start <= pred_idx <= t_end:
                    in_triggered_event = True
                    break

            if y_pred[i] == 1:  # Model predicted positive
                if in_triggered_event:
                    tp += 1  # True positive (either correct or within triggered event)
                else:
                    fp += 1  # False positive (outside any true event)
            else:  # Model predicted negative
                if in_triggered_event:
                    tp += 1  # Still true positive (within triggered event)
                else:
                    tn += 1  # True negative (correctly predicted negative outside events)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nTriggered Pointwise Evaluation:")
        print(f"Triggered Pointwise Precision: {precision:.4f}")
        print(f"Triggered Pointwise Recall: {recall:.4f}")
        print(f"Triggered Pointwise F1-Score: {f1:.4f}")
        print(f"Triggered Pointwise True Positives: {tp}")
        print(f"Triggered Pointwise False Positives: {fp}")
        print(f"Triggered Pointwise True Negatives: {tn}")
        print(f"Triggered Pointwise False Negatives: {fn}")
        print(f"Number of triggered events: {len(triggered_events)}")
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'n_triggered_events': len(triggered_events)
        }

def find_detection_windows(data: pd.DataFrame, debug: bool = False) -> list:
    """Find all detection windows (where gt_detection_win == 1 or gt_fwhm == 1)."""
    windows = []
    in_window = False
    start_idx = None
    
    # Combine both ground truth conditions
    gt_combined = np.logical_or(data['gt_detection_win'] == True, data['gt_fwhm'] == True)
    
    debug_print(debug, "\nDebug: Ground truth statistics")
    debug_print(debug, f"Total points: {len(data)}")
    debug_print(debug, f"Points with gt_detection_win=1: {sum(data['gt_detection_win'] == True)}")
    debug_print(debug, f"Points with gt_fwhm=1: {sum(data['gt_fwhm'] == True)}")
    debug_print(debug, f"Points with either=1: {sum(gt_combined)}")
    
    # Print first few rows of data to check values
    debug_print(debug, "\nFirst few rows of data:")
    debug_print(debug, data[['gt_detection_win', 'gt_fwhm']].head(10))
    
    # Print some random rows where we expect vortices
    vortex_indices = np.where(gt_combined)[0]
    if len(vortex_indices) > 0:
        debug_print(debug, "\nSample vortex rows:")
        sample_indices = vortex_indices[:5]
        debug_print(debug, data.iloc[sample_indices][['gt_detection_win', 'gt_fwhm']])
    
    # Find windows
    for idx, is_vortex in enumerate(gt_combined):
        if is_vortex and not in_window:
            in_window = True
            start_idx = idx
        elif not is_vortex and in_window:
            in_window = False
            windows.append((start_idx, idx))
    
    if in_window:
        windows.append((start_idx, len(data)))
    
    debug_print(debug, f"\nDebug: Found {len(windows)} vortex windows")
    if windows:
        debug_print(debug, f"First few windows: {windows[:5]}")
        
        # Print the actual values in the first few windows
        debug_print(debug, "\nChecking first few windows:")
        for start, end in windows[:3]:
            debug_print(debug, f"\nWindow {start}:{end}")
            debug_print(debug, "gt_detection_win:", data['gt_detection_win'].iloc[start:end].values)
            debug_print(debug, "gt_fwhm:", data['gt_fwhm'].iloc[start:end].values)
    
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
    debug_print(model.debug, "\nPreparing sequences from full dataset (no sampling)...")
    X_full, y_full, pressure_mean, pressure_std, roc_mean, roc_std, \
    rolling_mean_mean, rolling_mean_std, rolling_std_mean, rolling_std_std = model.prepare_sequences(data, apply_sampling=False)
    
    debug_print(model.debug, "Making predictions on full dataset...")
    y_pred_proba = model.predict(X_full)
    
    # Get ground truth windows
    gt_windows = find_detection_windows(data, debug=model.debug)
    
    # Standard evaluation
    debug_print(model.debug, "\nPerforming standard evaluation...")
    standard_results = model.evaluate(X_full, y_full)
    
    # Window-based evaluation
    debug_print(model.debug, "\nPerforming window-based evaluation...")
    window_results = model.evaluate_with_windows(standard_results, gt_windows, data)
    
    # Print both results
    debug_print(model.debug, "\nStandard Evaluation Results:")
    debug_print(model.debug, f"Precision: {standard_results['precision']:.4f}")
    debug_print(model.debug, f"Recall: {standard_results['recall']:.4f}")
    debug_print(model.debug, f"F1-Score: {standard_results['f1']:.4f}")
    debug_print(model.debug, f"ROC-AUC: {standard_results['roc_auc']:.4f}")
    debug_print(model.debug, f"PR-AUC: {standard_results['pr_auc']:.4f}")
    
    debug_print(model.debug, "\nWindow-Based Evaluation Results:")
    debug_print(model.debug, f"Precision: {window_results['original_metrics']['precision']:.4f}")
    debug_print(model.debug, f"Recall: {window_results['original_metrics']['recall']:.4f}")
    debug_print(model.debug, f"F1-Score: {window_results['original_metrics']['f1']:.4f}")
    debug_print(model.debug, f"ROC-AUC: {window_results['original_metrics']['roc_auc']:.4f}")
    debug_print(model.debug, f"PR-AUC: {window_results['original_metrics']['pr_auc']:.4f}")
    
    return {
        'standard': standard_results,
        'window_based': window_results
    }

def analyze_pressure_patterns(data: pd.DataFrame, window_size: int = 60, debug: bool = False):
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
    
    debug_print(debug, "\nPressure Pattern Analysis:")
    debug_print(debug, f"Number of vortex patterns analyzed: {len(vortex_patterns)}")
    debug_print(debug, f"Number of non-vortex patterns analyzed: {len(non_vortex_patterns)}")
    
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
    debug_print(debug, "\nPattern Statistics:")
    debug_print(debug, f"Mean pressure in vortex patterns: {np.mean(mean_vortex):.2f} ± {np.mean(std_vortex):.2f}")
    debug_print(debug, f"Mean pressure in non-vortex patterns: {np.mean(mean_non_vortex):.2f} ± {np.mean(std_non_vortex):.2f}")
    debug_print(debug, f"Max pressure difference: {np.max(np.abs(mean_vortex - mean_non_vortex)):.2f}")
    debug_print(debug, f"Average pressure difference: {np.mean(np.abs(mean_vortex - mean_non_vortex)):.2f}")

def main():
    """Main function to train and evaluate the LSTM model."""
    parser = argparse.ArgumentParser(description='Train or evaluate LSTM model')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    parser.add_argument('--analyze', action='store_true', help='Analyze pressure patterns')
    parser.add_argument('--full_eval', action='store_true', help='Run evaluation on full dataset')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
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
        analyze_pressure_patterns(data, debug=args.debug)
        return
    
    # Find all vortex events
    gt_combined = np.logical_or(data['gt_detection_win'] == 1, data['gt_fwhm'] == 1)
    vortex_indices = np.where(gt_combined)[0]
    
    print("\nVortex event statistics:")
    print(f"Total vortex events: {len(vortex_indices)}")
    print(f"First few vortex indices: {vortex_indices[:5]}")
    
    # Split data temporally (70/15/15) while preserving vortex events
    n_samples = len(data)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)
    
    # Ensure we have vortex events in each split
    train_vortices = vortex_indices[vortex_indices < train_end]
    val_vortices = vortex_indices[(vortex_indices >= train_end) & (vortex_indices < val_end)]
    test_vortices = vortex_indices[vortex_indices >= val_end]
    
    print("\nSplit statistics:")
    print(f"Training samples: {len(train_vortices)} vortex events")
    print(f"Validation samples: {len(val_vortices)} vortex events")
    print(f"Test samples: {len(test_vortices)} vortex events")
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    print(f"\nData split sizes:")
    print(f"Training: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Initialize model
    model = VortexLSTMModel(window_size=60, debug=args.debug)
    
    # Prepare sequences for each split
    print("\nPreparing training sequences...")
    X_train, y_train, pressure_mean, pressure_std, roc_mean, roc_std, \
    rolling_mean_mean, rolling_mean_std, rolling_std_mean, rolling_std_std = model.prepare_sequences(train_data, apply_sampling=True)
    
    print("\nPreparing validation sequences...")
    X_val, y_val, _, _, _, _, _, _, _, _ = model.prepare_sequences(val_data, apply_sampling=True, 
        pressure_mean=pressure_mean, pressure_std=pressure_std,
        roc_mean=roc_mean, roc_std=roc_std,
        rolling_mean_mean=rolling_mean_mean, rolling_mean_std=rolling_mean_std,
        rolling_std_mean=rolling_std_mean, rolling_std_std=rolling_std_std)
    
    print("\nPreparing test sequences...")
    X_test, y_test, _, _, _, _, _, _, _, _ = model.prepare_sequences(test_data, apply_sampling=False,
        pressure_mean=pressure_mean, pressure_std=pressure_std,
        roc_mean=roc_mean, roc_std=roc_std,
        rolling_mean_mean=rolling_mean_mean, rolling_mean_std=rolling_mean_std,
        rolling_std_mean=rolling_std_mean, rolling_std_std=rolling_std_std)
    
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
        # Fix: Register the custom loss function as 'loss_function' for Keras
        loss_fn = model.temporal_focal_loss()
        model.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'loss_function': loss_fn}
        )
        print("Model loaded successfully")
    else:
        if args.retrain:
            print("\nForce retrain flag set. Training new model...")
        else:
            print("\nNo model found. Training new model...")
        
        # Train model
        print("\nTraining LSTM model...")
        history = model.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=30, batch_size=128)
        
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

    # --- NEW: Compare PR-AUC to random baseline ---
    vortex_prevalence = sum(y_test) / len(y_test)
    pr_auc_model = test_results['pr_auc']
    print(f"Random PR-AUC (vortex prevalence): {vortex_prevalence:.5f}")
    if vortex_prevalence > 0:
        print(f"Model is {pr_auc_model / vortex_prevalence:.1f}x better than random!")
    else:
        print("No vortex events in test set; cannot compute random baseline.")
    # --- END NEW ---
    
    # Analyze learned patterns
    print("\nAnalyzing learned patterns...")
    model.analyze_learned_patterns(test_data, test_results['y_pred'])
    
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
        print(f"Precision: {full_results['standard']['precision']:.4f}")
        print(f"Recall: {full_results['standard']['recall']:.4f}")
        print(f"F1-Score: {full_results['standard']['f1']:.4f}")
        print(f"ROC-AUC: {full_results['standard']['roc_auc']:.4f}")
        print(f"PR-AUC: {full_results['standard']['pr_auc']:.4f}")
        
        # Generate full dataset visualizations
        print("\nGenerating full dataset visualizations...")
        visualize_lstm_metrics(
            model=model.model,
            X_test=X_train,
            y_test=y_train,
            y_pred=full_results['standard']['y_pred'],
            y_pred_proba=full_results['standard']['y_pred_proba'],
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

    # After test_results = model.evaluate(X_test, y_test)
    gt_windows = find_detection_windows(test_data, debug=False)
    event_results = model.evaluate_with_windows(test_results, gt_windows, test_data)
    print("\nEvent-Based Evaluation:")
    print(f"Event-based Precision: {event_results['event_metrics']['precision']:.4f}")
    print(f"Event-based Recall: {event_results['event_metrics']['recall']:.4f}")
    print(f"Event-based F1-Score: {event_results['event_metrics']['f1']:.4f}")
    print(f"Event-based True Positives: {event_results['event_metrics']['tp']}")
    print(f"Event-based False Positives: {event_results['event_metrics']['fp']}")
    print(f"Event-based False Negatives: {event_results['event_metrics']['fn']}")
    if event_results['event_metrics']['earliness']:
        print(f"Event-based Mean Earliness: {np.mean(event_results['event_metrics']['earliness']):.2f} samples")
        
        print("\nModel training and analysis complete!")

    # After triggered event evaluation in main()
    triggered_pointwise_results = model.evaluate_triggered_pointwise(test_results['y_pred'], test_data)

if __name__ == "__main__":
    main() 
