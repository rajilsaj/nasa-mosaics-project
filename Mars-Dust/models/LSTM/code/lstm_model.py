import pandas as pd
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> origin/2Stage_LSTM
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
<<<<<<< HEAD
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



=======
from pathlib import Path
import time
import argparse
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import numpy as np
from evaluation_utils import compute_classification_metrics, sweep_confidence_thresholds
from plotting_utils import (plot_confidence_distribution, plot_confidence_timeline, 
                           plot_detection_patterns, plot_confidence_analysis, 
                           plot_pressure_patterns, plot_training_history)

# Import artifact detector
from artifact_detector import ArtifactDetector
print("Using GPU:", tf.config.list_physical_devices('GPU'))
mixed_precision.set_global_policy("mixed_float16")

>>>>>>> origin/2Stage_LSTM
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
        
<<<<<<< HEAD
    def prepare_sequences(self, data: pd.DataFrame, apply_sampling: bool = True, feature_set: str = 'shape') -> tuple:
=======
    def prepare_sequences(self, data: pd.DataFrame, apply_sampling: bool = True, 
                         use_artifacts: bool = False, artifact_ratio: float = 0.5) -> tuple:
>>>>>>> origin/2Stage_LSTM
        """
        Prepare sequences for vortex prediction using local detrending.
        Each window is normalized independently to make features invariant
        to the absolute pressure baseline, forcing the model to learn the
        shape of the pressure curve.
        
        Args:
            data: Input data
            apply_sampling: If True, creates balanced dataset from vortex events.
                          If False, processes all available data using a sliding window.
<<<<<<< HEAD
            feature_set: 'shape' for (pressure, acceleration) or 'statistical' for (pressure, skew, kurtosis)
=======
            use_artifacts: If True, include artifact windows as negative training examples
            artifact_ratio: Ratio of artifacts to include relative to vortex events
>>>>>>> origin/2Stage_LSTM
        """
        pressure_values = data['PRESSURE'].values
        gt_detection = data['gt_detection_win'].values
        gt_fwhm = data['gt_fwhm'].values
        
        sequences_list = []
        labels_list = []

        if not apply_sampling:
            # --- Process all data with a sliding window for testing/evaluation ---
            self.debug_print("\nProcessing all available data for test/evaluation using sliding window with local detrending...")
            for i in range(self.window_size, len(data)):
                # 1. Get the raw pressure window
                pressure_window = pressure_values[i-self.window_size:i].copy()
                
                # 2. Detrend by subtracting the window's own mean
                local_mean = np.mean(pressure_window)
                detrended_pressure = pressure_window - local_mean
                
<<<<<<< HEAD
                # 3. Calculate features based on the chosen set
                if feature_set == 'shape':
                    acceleration = np.diff(detrended_pressure, n=2, prepend=[detrended_pressure[0], detrended_pressure[0]])
                    sequence = np.stack([detrended_pressure, acceleration], axis=1)
                elif feature_set == 'statistical':
                    rolling_series = pd.Series(detrended_pressure)
                    skewness = rolling_series.rolling(window=20, min_periods=1).skew().fillna(0).values
                    kurtosis = rolling_series.rolling(window=20, min_periods=1).kurt().fillna(0).values
                    sequence = np.stack([detrended_pressure, skewness, kurtosis], axis=1)
                else:
                    raise ValueError("feature_set must be 'shape' or 'statistical'")
                
                # 6. Determine the label for the window
=======
                # 3. Reshape for LSTM: (window_size, 1) - single feature
                sequence = detrended_pressure.reshape(-1, 1)
                
                # 4. Determine the label for the window
>>>>>>> origin/2Stage_LSTM
                label = 1 if np.any(gt_detection[i-self.window_size:i] == 1) else 0
                sequences_list.append(sequence)
                labels_list.append(label)

            self.debug_print("\nFull data statistics (after local detrending):")
            self.debug_print(f"Total sequences: {len(sequences_list)}")
            self.debug_print(f"Vortex windows: {sum(labels_list)}")

        else:
            # --- Create a balanced dataset for training/validation ---
            self.debug_print("\nCreating balanced dataset for training/validation with local detrending...")
            vortex_indices = np.where(np.logical_or(gt_detection == 1, gt_fwhm == 1))[0]
            
            for vortex_idx in vortex_indices:
                if vortex_idx >= self.window_size:
                    # Create a positive sample (the window ending at the vortex)
                    pressure_window = pressure_values[vortex_idx-self.window_size:vortex_idx].copy()
                    local_mean = np.mean(pressure_window)
                    detrended_pressure = pressure_window - local_mean
                    
<<<<<<< HEAD
                    if feature_set == 'shape':
                        acceleration = np.diff(detrended_pressure, n=2, prepend=[detrended_pressure[0], detrended_pressure[0]])
                        sequence = np.stack([detrended_pressure, acceleration], axis=1)
                    elif feature_set == 'statistical':
                        rolling_series = pd.Series(detrended_pressure)
                        skewness = rolling_series.rolling(window=20, min_periods=1).skew().fillna(0).values
                        kurtosis = rolling_series.rolling(window=20, min_periods=1).kurt().fillna(0).values
                        sequence = np.stack([detrended_pressure, skewness, kurtosis], axis=1)

=======
                    # Reshape for LSTM: (window_size, 1) - single feature
                    sequence = detrended_pressure.reshape(-1, 1)
>>>>>>> origin/2Stage_LSTM
                    sequences_list.append(sequence)
                    labels_list.append(1)

                    # Create a negative sample (a random window from a non-vortex area)
                    # This helps the model learn to distinguish vortex shapes from other noise
                    while True:
                        random_idx = np.random.randint(self.window_size, len(pressure_values))
                        if not np.any(gt_detection[random_idx-self.window_size:random_idx] == 1):
                            break
                    
                    pressure_window_neg = pressure_values[random_idx-self.window_size:random_idx].copy()
                    local_mean_neg = np.mean(pressure_window_neg)
                    detrended_pressure_neg = pressure_window_neg - local_mean_neg
                    
<<<<<<< HEAD
                    if feature_set == 'shape':
                        acceleration_neg = np.diff(detrended_pressure_neg, n=2, prepend=[detrended_pressure_neg[0], detrended_pressure_neg[0]])
                        sequence_neg = np.stack([detrended_pressure_neg, acceleration_neg], axis=1)
                    elif feature_set == 'statistical':
                        rolling_series_neg = pd.Series(detrended_pressure_neg)
                        skewness_neg = rolling_series_neg.rolling(window=20, min_periods=1).skew().fillna(0).values
                        kurtosis_neg = rolling_series_neg.rolling(window=20, min_periods=1).kurt().fillna(0).values
                        sequence_neg = np.stack([detrended_pressure_neg, skewness_neg, kurtosis_neg], axis=1)

=======
                    # Reshape for LSTM: (window_size, 1) - single feature
                    sequence_neg = detrended_pressure_neg.reshape(-1, 1)
>>>>>>> origin/2Stage_LSTM
                    sequences_list.append(sequence_neg)
                    labels_list.append(0)

            self.debug_print("\nBalanced data statistics (after local detrending):")
            self.debug_print(f"Total sequences: {len(sequences_list)}")
            self.debug_print(f"Vortex sequences: {sum(labels_list)}")
<<<<<<< HEAD
=======
            
            # Add artifact windows if requested
            if use_artifacts:
                self.debug_print("\nAdding artifact windows as negative training examples...")
                artifact_detector = ArtifactDetector(window_size=self.window_size)
                X_artifacts, y_artifacts = artifact_detector.prepare_artifact_training_data(data, artifact_ratio)
                
                if len(X_artifacts) > 0:
                    # Convert artifact sequences to match simplified format (single feature)
                    artifact_sequences = []
                    for artifact in X_artifacts:
                        # Artifacts are already detrended pressure, just reshape to single feature
                        detrended_pressure = artifact[:, 0]  # First channel is pressure
                        artifact_single_feature = detrended_pressure.reshape(-1, 1)
                        artifact_sequences.append(artifact_single_feature)
                    
                    # Add artifacts to sequences
                    sequences_list.extend(artifact_sequences)
                    labels_list.extend(y_artifacts)
                    
                    self.debug_print(f"Added {len(artifact_sequences)} artifact sequences")
                    self.debug_print(f"Updated total sequences: {len(sequences_list)}")
                    self.debug_print(f"Updated vortex sequences: {sum(labels_list)}")
                else:
                    self.debug_print("No artifacts detected in this dataset")
>>>>>>> origin/2Stage_LSTM

        sequences = np.array(sequences_list)
        labels = np.array(labels_list)
        
<<<<<<< HEAD
        # We no longer return normalization stats
=======
>>>>>>> origin/2Stage_LSTM
        return sequences, labels
        
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
    
<<<<<<< HEAD
    def build_model(self, input_shape: tuple, alpha: float = None, gamma: float = 1.5):
=======
    def build_model(self, input_shape: tuple, alpha: float = None, gamma: float = 1.5, learning_rate: float = 0.01):
>>>>>>> origin/2Stage_LSTM
        """Build the vortex prediction model with Bidirectional LSTM and temporal loss."""
        model = Sequential([
            Bidirectional(
                LSTM(64,  # Adjusted units for fewer features
                     kernel_regularizer=l2(0.0005),
                     recurrent_regularizer=l2(0.0005),
                     return_sequences=False),
                input_shape=input_shape
            ),
            Dense(32, activation='relu'), # Adjusted units
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
<<<<<<< HEAD
            optimizer=Adam(learning_rate=0.001),
=======
            optimizer=Adam(learning_rate=learning_rate),
>>>>>>> origin/2Stage_LSTM
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

<<<<<<< HEAD
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, batch_size=256):
=======
    def train(self, X_train, y_train, X_val, y_val, X_test, y_test, epochs=50, batch_size=256, learning_rate=0.01):
>>>>>>> origin/2Stage_LSTM
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
        # The input shape will now depend on the feature set
        num_features = X_train.shape[2]
<<<<<<< HEAD
        self.model = self.build_model((self.window_size, num_features), alpha=alpha, gamma=1.5)
=======
        self.model = self.build_model((self.window_size, num_features), alpha=alpha, gamma=1.5, learning_rate=learning_rate)
>>>>>>> origin/2Stage_LSTM
        
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
    
<<<<<<< HEAD
    def predict_real_time(self, pressure_readings: np.ndarray) -> float:
        """Make real-time predictions on new pressure readings."""
        if len(pressure_readings) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} pressure readings")
            
        current_readings = pressure_readings[-self.window_size:]
        # Note: This real-time function will need to know which features to build.
        # This is a simplification for now.
        sequence = current_readings.reshape(1, self.window_size, current_readings.shape[1])
        return self.model.predict(sequence)[0][0]
    
=======
>>>>>>> origin/2Stage_LSTM
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

    def evaluate_event_level(self, y_pred, window_starts, window_size, events, test_data):
        """
        Evaluate predictions at the event level.
        A single positive prediction is a "detection".
        A true event is considered "detected" if any detection falls within its range.
        """
        fp = 0
        # A set to store the indices of true events that have been detected.
        # This prevents double-counting a single true event.
        detected_true_events = set()
        
        # Create a quick lookup for true events for efficiency
        # This maps each index to the event_id it belongs to.
        point_to_event_map = {}
        for event_idx, (start, end) in enumerate(events):
            for i in range(start, end + 1):
                point_to_event_map[i] = event_idx

        # --- SCLK SANITY CHECK ---
        print("\n[SCLK CHECK] True Event SCLK Ranges:")
        for event_idx, (start, end) in enumerate(events[:5]): # Print first 5 for brevity
            sclk_start = test_data['SCLK'].iloc[start]
            sclk_end = test_data['SCLK'].iloc[end]
            print(f"  - True Event {event_idx}: Index [{start}, {end}] -> SCLK [{sclk_start}, {sclk_end}]")
        # --- END SCLK SANITY CHECK ---

        # --- DEBUG PRINTS ---
        print("\n[DEBUG] Using single-point detection logic.")
        print(f"[DEBUG] Total positive predictions (potential detections): {np.sum(y_pred)}")
        print(f"[DEBUG] Number of true events: {len(events)}")
        # --- END DEBUG PRINTS ---

        # Iterate through each prediction once
        positive_prediction_count = 0
        for i, prediction in enumerate(y_pred):
            if prediction == 1:
                positive_prediction_count += 1
                # This is a positive prediction, a "detection"
                detection_idx = window_starts[i]

                # --- SCLK SANITY CHECK (limited) ---
                if positive_prediction_count <= 25:
                    detection_sclk = test_data['SCLK'].iloc[detection_idx]
                    print(f"[SCLK CHECK] Positive prediction #{positive_prediction_count} at pred_index={i} -> data_index={detection_idx} -> SCLK={detection_sclk}")
                # --- END SCLK SANITY CHECK ---

                # Check if this detection index corresponds to any true event
                if detection_idx in point_to_event_map:
                    # It's a true positive detection, mark the corresponding event as detected
                    event_id = point_to_event_map[detection_idx]
                    detected_true_events.add(event_id)
                else:
                    # This detection does not fall into any true event range, so it's a false positive.
                    fp += 1
        
        # True Positives is the number of unique true events that were detected.
        tp = len(detected_true_events)
        
        # False Negatives is the number of true events that were not detected.
        fn = len(events) - tp

        # Earliness is not relevant in this model.
        earliness = []

        print(f"[DEBUG] Number of TPs (unique detected events): {tp}")
        print(f"[DEBUG] Number of FPs (false alarms): {fp}")
        print(f"[DEBUG] Number of FNs (missed events): {fn}")

        return tp, fp, fn, earliness

    def analyze_detection_patterns(self, y_pred, window_starts, events, test_data):
        """Analyze pressure patterns around successful vs failed detections."""
<<<<<<< HEAD
        import matplotlib.pyplot as plt
=======
        # import matplotlib.pyplot as plt  # Moved to plotting_utils.py
>>>>>>> origin/2Stage_LSTM
        
        # Create lookup for true events
        point_to_event_map = {}
        for event_idx, (start, end) in enumerate(events):
            for i in range(start, end + 1):
                point_to_event_map[i] = event_idx
        
        # Collect patterns
        successful_patterns = []
        failed_patterns = []
        successful_sclks = []
        failed_sclks = []
        
<<<<<<< HEAD
        window_size = 60  # Look at 60 points before and after detection
        
=======
        window_size = self.window_size
>>>>>>> origin/2Stage_LSTM
        for i, prediction in enumerate(y_pred):
            if prediction == 1:  # Positive prediction
                detection_idx = window_starts[i]
                detection_sclk = test_data['SCLK'].iloc[detection_idx]
<<<<<<< HEAD
                
                # Get pressure pattern around this detection
                start_idx = max(0, detection_idx - window_size)
                end_idx = min(len(test_data), detection_idx + window_size + 1)
                pressure_pattern = test_data['PRESSURE'].iloc[start_idx:end_idx].values
                
                if detection_idx in point_to_event_map:
                    # Successful detection
                    successful_patterns.append(pressure_pattern)
                    successful_sclks.append(detection_sclk)
                else:
                    # False alarm
                    failed_patterns.append(pressure_pattern)
                    failed_sclks.append(detection_sclk)
        
        # Convert to arrays
=======
                # Get pressure pattern ending at detection (causal)
                start_idx = max(0, detection_idx - window_size)
                end_idx = detection_idx + 1  # inclusive of detection point
                pressure_pattern = test_data['PRESSURE'].iloc[start_idx:end_idx].values
                if len(pressure_pattern) == window_size + 1:
                    if detection_idx in point_to_event_map:
                        successful_patterns.append(pressure_pattern)
                        successful_sclks.append(detection_sclk)
                    else:
                        failed_patterns.append(pressure_pattern)
                        failed_sclks.append(detection_sclk)
        # Update time_points for plotting to np.arange(-window_size, 1)
        time_points = np.arange(-window_size, 1)
        # All subsequent pattern analysis, continued drop analysis, and plots should use these causal patterns and time_points.
        # In continued drop analysis, use only these causal patterns as well.
        
        # Convert to arrays (after all appends)
>>>>>>> origin/2Stage_LSTM
        successful_patterns = np.array(successful_patterns)
        failed_patterns = np.array(failed_patterns)
        
        print(f"\n[PATTERN ANALYSIS] Successful detections: {len(successful_patterns)}")
        print(f"[PATTERN ANALYSIS] False alarms: {len(failed_patterns)}")
        
<<<<<<< HEAD
=======
        # Analyze statistical differences between patterns
        if len(successful_patterns) > 0 and len(failed_patterns) > 0:
            from pattern_analysis import analyze_pattern_statistics, find_best_thresholds
            
            print("\n[PATTERN ANALYSIS] Analyzing statistical differences...")
            stats = analyze_pattern_statistics(successful_patterns, failed_patterns)
            thresholds = find_best_thresholds(stats)
            
            # Store the best discriminative features for potential use in artifact detection
            if thresholds:
                best_feature = max(thresholds.keys(), key=lambda k: thresholds[k]['f1_score'])
                best_f1 = thresholds[best_feature]['f1_score']
                print(f"\n[PATTERN ANALYSIS] Best discriminative feature: {best_feature} (F1={best_f1:.3f})")
                print(f"[PATTERN ANALYSIS] This could be used to improve artifact detection!")
        
>>>>>>> origin/2Stage_LSTM
        if len(successful_patterns) > 0 and len(failed_patterns) > 0:
            # Calculate statistics
            mean_successful = np.mean(successful_patterns, axis=0)
            mean_failed = np.mean(failed_patterns, axis=0)
            std_successful = np.std(successful_patterns, axis=0)
            std_failed = np.std(failed_patterns, axis=0)
            
            print(f"[PATTERN ANALYSIS] Mean pressure in successful patterns: {np.mean(mean_successful):.2f} ± {np.mean(std_successful):.2f}")
            print(f"[PATTERN ANALYSIS] Mean pressure in failed patterns: {np.mean(mean_failed):.2f} ± {np.mean(std_failed):.2f}")
            print(f"[PATTERN ANALYSIS] Pressure difference: {np.mean(mean_successful - mean_failed):.2f}")
            
<<<<<<< HEAD
            # Plot patterns
            plt.figure(figsize=(15, 10))
            
            # Plot mean patterns
            plt.subplot(2, 2, 1)
            time_points = np.arange(-window_size, window_size + 1)
            plt.plot(time_points, mean_successful, label='Successful Detections', color='green', linewidth=2)
            plt.plot(time_points, mean_failed, label='False Alarms', color='red', linewidth=2)
            plt.fill_between(time_points, mean_successful - std_successful, mean_successful + std_successful, 
                           color='green', alpha=0.2)
            plt.fill_between(time_points, mean_failed - std_failed, mean_failed + std_failed, 
                           color='red', alpha=0.2)
            plt.title('Mean Pressure Patterns Around Detections')
            plt.xlabel('Time Points (relative to detection)')
            plt.ylabel('Pressure')
            plt.legend()
            plt.grid(True)
            
            # Plot difference
            plt.subplot(2, 2, 2)
            plt.plot(time_points, mean_successful - mean_failed, color='blue', linewidth=2)
            plt.title('Difference (Successful - Failed)')
            plt.xlabel('Time Points (relative to detection)')
            plt.ylabel('Pressure Difference')
            plt.grid(True)
            
            # Plot individual patterns (first few)
            plt.subplot(2, 2, 3)
            for i in range(min(5, len(successful_patterns))):
                plt.plot(time_points, successful_patterns[i], color='green', alpha=0.3)
            for i in range(min(5, len(failed_patterns))):
                plt.plot(time_points, failed_patterns[i], color='red', alpha=0.3)
            plt.title('Individual Patterns (First 5 of each)')
            plt.xlabel('Time Points (relative to detection)')
            plt.ylabel('Pressure')
            plt.grid(True)
            
            # Plot SCLK distribution
            plt.subplot(2, 2, 4)
            if len(successful_sclks) > 0:
                plt.hist(successful_sclks, bins=20, alpha=0.7, label='Successful', color='green')
            if len(failed_sclks) > 0:
                plt.hist(failed_sclks, bins=50, alpha=0.7, label='Failed', color='red')
            plt.title('SCLK Distribution of Detections')
            plt.xlabel('SCLK')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('detection_patterns.png')
            plt.close()
=======
            # Calculate minimum slope (sharpest drop) for each pattern
            min_slope_successful = [np.min(np.diff(p)) for p in successful_patterns]
            min_slope_failed = [np.min(np.diff(p)) for p in failed_patterns]

            # 6. Sharpest short-window slope (max negative difference over 3 consecutive points)
            def max_short_window_slope(pattern, window=3):
                return np.min([pattern[i+window-1] - pattern[i] for i in range(len(pattern)-window+1)])
            short_window_slope_successful = [max_short_window_slope(p, window=3) for p in successful_patterns]
            short_window_slope_failed = [max_short_window_slope(p, window=3) for p in failed_patterns]

            # 7. Drop duration (number of points from start to end of main drop)
            def drop_duration(pattern, threshold=0.1):
                # Find the largest drop in the pattern
                diffs = np.diff(pattern)
                min_idx = np.argmin(diffs)
                # Go backwards to find where the drop started
                start = min_idx
                while start > 0 and diffs[start] < -threshold:
                    start -= 1
                # Go forwards to find where the drop ended
                end = min_idx
                while end < len(diffs)-1 and diffs[end] < -threshold:
                    end += 1
                return end - start + 1

            # Use a threshold of 0.1 for what counts as a drop
            sharp_drop_duration_successful = [drop_duration(p, threshold=0.1) for p in successful_patterns]
            sharp_drop_duration_failed = [drop_duration(p, threshold=0.1) for p in failed_patterns]

            # Plot detection patterns using the plotting module
            plot_detection_patterns(successful_patterns, failed_patterns, successful_sclks, failed_sclks,
                                  min_slope_successful, min_slope_failed, short_window_slope_successful, 
                                  short_window_slope_failed, sharp_drop_duration_successful, sharp_drop_duration_failed,
                                  time_points, mean_successful, mean_failed, std_successful, std_failed)
>>>>>>> origin/2Stage_LSTM
            
            print(f"[PATTERN ANALYSIS] Pattern analysis saved to detection_patterns.png")
        
        return {
            'successful_patterns': successful_patterns,
            'failed_patterns': failed_patterns,
            'successful_sclks': successful_sclks,
            'failed_sclks': failed_sclks
        }

    def analyze_confidence_distribution(self, y_pred, y_pred_proba, window_starts, events, test_data):
        """Analyze confidence scores for successful vs failed detections."""
<<<<<<< HEAD
        import matplotlib.pyplot as plt
=======
        # import matplotlib.pyplot as plt  # Moved to plotting_utils.py
>>>>>>> origin/2Stage_LSTM
        
        # Create lookup for true events
        point_to_event_map = {}
        for event_idx, (start, end) in enumerate(events):
            for i in range(start, end + 1):
                point_to_event_map[i] = event_idx
        
        # Collect confidence scores
        successful_confidences = []
        failed_confidences = []
        
        for i, (prediction, confidence) in enumerate(zip(y_pred, y_pred_proba)):
            if prediction == 1:  # Positive prediction
                detection_idx = window_starts[i]
                
                if detection_idx in point_to_event_map:
                    # Successful detection
                    successful_confidences.append(confidence)
                else:
                    # False alarm
                    failed_confidences.append(confidence)
        
        successful_confidences = np.array(successful_confidences)
        failed_confidences = np.array(failed_confidences)
        
        print(f"\n[CONFIDENCE ANALYSIS] Successful detections: {len(successful_confidences)}")
        print(f"[CONFIDENCE ANALYSIS] False alarms: {len(failed_confidences)}")
        
        if len(successful_confidences) > 0:
            print(f"[CONFIDENCE ANALYSIS] Successful confidence - Mean: {np.mean(successful_confidences):.4f}, Std: {np.std(successful_confidences):.4f}")
        if len(failed_confidences) > 0:
            print(f"[CONFIDENCE ANALYSIS] Failed confidence - Mean: {np.mean(failed_confidences):.4f}, Std: {np.std(failed_confidences):.4f}")
        
        if len(successful_confidences) > 0 and len(failed_confidences) > 0:
            print(f"[CONFIDENCE ANALYSIS] Confidence difference: {np.mean(successful_confidences) - np.mean(failed_confidences):.4f}")
            
<<<<<<< HEAD
            # Plot confidence distributions
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.hist(successful_confidences, bins=20, alpha=0.7, label='Successful', color='green', density=True)
            plt.hist(failed_confidences, bins=50, alpha=0.7, label='Failed', color='red', density=True)
            plt.title('Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            
            # Plot confidence vs threshold analysis
            plt.subplot(1, 2, 2)
            thresholds = np.linspace(0.1, 0.9, 81)
            successful_counts = []
            failed_counts = []
            
            for threshold in thresholds:
                successful_counts.append(np.sum(successful_confidences >= threshold))
                failed_counts.append(np.sum(failed_confidences >= threshold))
            
            plt.plot(thresholds, successful_counts, label='Successful', color='green', linewidth=2)
            plt.plot(thresholds, failed_counts, label='Failed', color='red', linewidth=2)
            plt.title('Detections vs Threshold')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Number of Detections')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('confidence_analysis.png')
            plt.close()
=======
            # Plot confidence analysis using the plotting module
            plot_confidence_analysis(successful_confidences, failed_confidences)
>>>>>>> origin/2Stage_LSTM
            
            print(f"[CONFIDENCE ANALYSIS] Confidence analysis saved to confidence_analysis.png")
            
            # Suggest optimal threshold
            if len(successful_confidences) > 0:
                # Find threshold that maximizes successful while minimizing failed
                best_ratio = 0
                best_threshold = 0.5
<<<<<<< HEAD
=======
                thresholds = np.linspace(0.1, 0.9, 81)
>>>>>>> origin/2Stage_LSTM
                
                for threshold in thresholds:
                    successful_above = np.sum(successful_confidences >= threshold)
                    failed_above = np.sum(failed_confidences >= threshold)
                    
                    if failed_above > 0:
                        ratio = successful_above / failed_above
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_threshold = threshold
                
                print(f"[CONFIDENCE ANALYSIS] Suggested threshold: {best_threshold:.3f} (ratio: {best_ratio:.2f})")
        
        return {
            'successful_confidences': successful_confidences,
            'failed_confidences': failed_confidences
        }

<<<<<<< HEAD
=======
    def evaluate_triggered_pointwise(self, y_pred, test_data):
        """
        Evaluate using a custom 'latch-on' pointwise logic.
        Once an event is 'triggered' by a positive prediction, all subsequent points
        within that true event are counted as True Positives.
        """
        # --- Setup ---
        gt_detection = test_data['gt_detection_win'].values
        gt_fwhm = test_data['gt_fwhm'].values
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        n_samples = len(gt_combined)
        window_size = self.window_size

        # Create a full-length prediction array aligned with the main data array
        aligned_y_pred = np.zeros(n_samples, dtype=int)
        # Predictions from the model correspond to the END of a window.
        # So a prediction at y_pred[i] corresponds to test_data index i + window_size - 1
        pred_indices = np.arange(len(y_pred)) + window_size - 1
        # Ensure we don't go out of bounds
        valid_indices = pred_indices < n_samples
        aligned_y_pred[pred_indices[valid_indices]] = y_pred[valid_indices]

        # 1. Identify all ground truth event windows
        true_events = self.extract_event_ranges(gt_detection, gt_fwhm)

        # 2. For each true event, find the index of the *first* positive prediction
        trigger_indices = {}  # Maps event_id -> trigger_index
        for event_idx, (start, end) in enumerate(true_events):
            first_trigger = -1
            # Look for a trigger within the event's span
            for i in range(start, end + 1):
                if aligned_y_pred[i] == 1:
                    first_trigger = i
                    break  # Found the first one
            trigger_indices[event_idx] = first_trigger # Will be -1 if not triggered

        # 3. Calculate metrics point-by-point based on the "latch-on" logic
        tp, fp, fn, tn = 0, 0, 0, 0
        point_to_event_map = {i: eid for eid, (start, end) in enumerate(true_events) for i in range(start, end + 1)}

        for i in range(n_samples):
            is_gt_positive = gt_combined[i]

            if is_gt_positive:
                event_id = point_to_event_map[i]
                trigger_idx = trigger_indices[event_id]

                if trigger_idx == -1:
                    # The event was never triggered, so this point is a False Negative
                    fn += 1
                else:
                    # The event was triggered at some point
                    if i < trigger_idx:
                        # This point is before the first trigger, so it's a miss
                        fn += 1
                    else:
                        # This point is at or after the trigger, count as a True Positive
                        tp += 1
            else:
                # This is a non-vortex point
                if aligned_y_pred[i] == 1:
                    fp += 1
                else:
                    tn += 1
                    
        # --- Final Metrics Calculation ---
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nTriggered Pointwise Evaluation (Corrected 'Latch-on' Logic):")
        print(f"Triggered Pointwise Precision: {precision:.4f}")
        print(f"Triggered Pointwise Recall: {recall:.4f}")
        print(f"Triggered Pointwise F1-Score: {f1:.4f}")
        print(f"Triggered Pointwise True Positives: {tp}")
        print(f"Triggered Pointwise False Positives: {fp}")
        print(f"Triggered Pointwise True Negatives: {tn}")
        print(f"Triggered Pointwise False Negatives: {fn}")
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }


>>>>>>> origin/2Stage_LSTM
    def evaluate_with_windows(self, test_results, gt_windows, test_data):
        """Evaluate model performance considering ground truth windows.
        
        Args:
            test_results: Dictionary containing original test results (y_pred, y_pred_proba, y_true)
            gt_windows: List of (start_idx, end_idx) tuples for ground truth windows
            test_data: Original test data DataFrame containing gt_detection_win
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
<<<<<<< HEAD
        import matplotlib.pyplot as plt
=======
        # import matplotlib.pyplot as plt  # Moved to plotting_utils.py
>>>>>>> origin/2Stage_LSTM
        
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
        tp, fp, fn, earliness = self.evaluate_event_level(y_pred, window_starts, self.window_size, events, test_data)
        
        # Run pattern analysis
        pattern_results = self.analyze_detection_patterns(y_pred, window_starts, events, test_data)
        
        # Run confidence analysis
        confidence_results = self.analyze_confidence_distribution(y_pred, y_pred_proba, window_starts, events, test_data)
        
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

<<<<<<< HEAD
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
        """
        Evaluate using a custom 'latch-on' pointwise logic.
        Once an event is 'triggered' by a positive prediction, all subsequent points
        within that true event are counted as True Positives.
        """
        # --- Setup ---
        gt_detection = test_data['gt_detection_win'].values
        gt_fwhm = test_data['gt_fwhm'].values
        gt_combined = np.logical_or(gt_detection == 1, gt_fwhm == 1)
        n_samples = len(gt_combined)
        window_size = self.window_size

        # Create a full-length prediction array aligned with the main data array
        aligned_y_pred = np.zeros(n_samples, dtype=int)
        # Predictions from the model correspond to the END of a window.
        # So a prediction at y_pred[i] corresponds to test_data index i + window_size - 1
        pred_indices = np.arange(len(y_pred)) + window_size -1
        # Ensure we don't go out of bounds
        valid_indices = pred_indices < n_samples
        aligned_y_pred[pred_indices[valid_indices]] = y_pred[valid_indices]

        # 1. Identify all ground truth event windows
        true_events = self.extract_event_ranges(gt_detection, gt_fwhm)

        # 2. For each true event, find the index of the *first* positive prediction
        trigger_indices = {}  # Maps event_id -> trigger_index
        for event_idx, (start, end) in enumerate(true_events):
            first_trigger = -1
            # Look for a trigger within the event's span
            for i in range(start, end + 1):
                if aligned_y_pred[i] == 1:
                    first_trigger = i
                    break  # Found the first one
            trigger_indices[event_idx] = first_trigger # Will be -1 if not triggered

        # 3. Calculate metrics point-by-point based on the "latch-on" logic
        tp, fp, fn, tn = 0, 0, 0, 0
        point_to_event_map = {i: eid for eid, (start, end) in enumerate(true_events) for i in range(start, end + 1)}

        for i in range(n_samples):
            is_gt_positive = gt_combined[i]

            if is_gt_positive:
                event_id = point_to_event_map[i]
                trigger_idx = trigger_indices[event_id]

                if trigger_idx == -1:
                    # The event was never triggered, so this point is a False Negative
                    fn += 1
                else:
                    # The event was triggered at some point
                    if i < trigger_idx:
                        # This point is before the first trigger, so it's a miss
                        fn += 1
                    else:
                        # This point is at or after the trigger, count as a True Positive
                        tp += 1
            else:
                # This is a non-vortex point
                if aligned_y_pred[i] == 1:
                    fp += 1
                else:
                    tn += 1
                    
        # --- Final Metrics Calculation ---
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nTriggered Pointwise Evaluation (Corrected 'Latch-on' Logic):")
        print(f"Triggered Pointwise Precision: {precision:.4f}")
        print(f"Triggered Pointwise Recall: {recall:.4f}")
        print(f"Triggered Pointwise F1-Score: {f1:.4f}")
        print(f"Triggered Pointwise True Positives: {tp}")
        print(f"Triggered Pointwise False Positives: {fp}")
        print(f"Triggered Pointwise True Negatives: {tn}")
        print(f"Triggered Pointwise False Negatives: {fn}")
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

=======
>>>>>>> origin/2Stage_LSTM
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

<<<<<<< HEAD
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

def evaluate_on_full_dataset(model: VortexLSTMModel, data: pd.DataFrame, feature_set: str = 'shape') -> dict:
    """Evaluate model performance on the full dataset."""
    debug_print(model.debug, "\nPreparing sequences from full dataset (no sampling)...")
    X_full, y_full = model.prepare_sequences(data, apply_sampling=False, feature_set=feature_set)
    
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
=======
def evaluate_on_full_dataset(model: VortexLSTMModel, data: pd.DataFrame) -> dict:
    """Evaluate model on the full dataset."""
    print("\nEvaluating on full dataset...")
    
    # Prepare sequences from full dataset
    X_full, y_full = model.prepare_sequences(data, apply_sampling=False)
    
    # Make predictions
    y_pred_proba = model.predict(X_full)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
    
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
    
    # Use the best threshold for final evaluation
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    precision = precision_score(y_full, y_pred)
    recall = recall_score(y_full, y_pred)
    f1 = f1_score(y_full, y_pred)
    roc_auc = roc_auc_score(y_full, y_pred_proba)
    pr_auc = average_precision_score(y_full, y_pred_proba)
    
    return {
        'standard': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'threshold': best_threshold,
            'y_true': y_full
        }
>>>>>>> origin/2Stage_LSTM
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
    
<<<<<<< HEAD
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
=======
    # Plot the patterns using the plotting module
    results_dir = Path(__file__).parent.parent / 'results'
    plot_pressure_patterns(mean_vortex, mean_non_vortex, std_vortex, std_non_vortex, results_dir)
>>>>>>> origin/2Stage_LSTM
    
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
<<<<<<< HEAD
    parser.add_argument('--window_size', type=int, default=30, help='Size of the input window for the LSTM.')
    parser.add_argument('--model_name', type=str, default='lstm_model.h5', help='Name for the saved model file.')
    parser.add_argument('--feature_set', type=str, default='shape', help="Feature set to use: 'shape' or 'statistical'")
=======
    parser.add_argument('--window_size', type=int, default=60, help='Size of the input window for the LSTM.')
    parser.add_argument('--model_name', type=str, default='lstm_model.h5', help='Name for the saved model file.')
    parser.add_argument('--use_artifacts', action='store_true', help='Include artifact windows as negative training examples')
    parser.add_argument('--artifact_ratio', type=float, default=0.5, help='Ratio of artifacts to include relative to vortex events (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training (default: 0.01)')
>>>>>>> origin/2Stage_LSTM
    args = parser.parse_args()
    
    print("Starting LSTM model training...")
    
<<<<<<< HEAD
=======
    # Show artifact configuration
    if args.use_artifacts:
        print(f"Artifact integration enabled with ratio: {args.artifact_ratio}")
    else:
        print("Artifact integration disabled")
    
>>>>>>> origin/2Stage_LSTM
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
<<<<<<< HEAD
        analyze_pressure_patterns(data, debug=args.debug)
=======
        analyze_pressure_patterns(data, window_size=args.window_size, debug=args.debug)
>>>>>>> origin/2Stage_LSTM
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
    model = VortexLSTMModel(window_size=args.window_size, debug=args.debug)
    
    # Prepare sequences for each split
<<<<<<< HEAD
    print(f"\nPreparing training sequences with '{args.feature_set}' features...")
    X_train, y_train = model.prepare_sequences(train_data, apply_sampling=True, feature_set=args.feature_set)
    
    print(f"\nPreparing validation sequences with '{args.feature_set}' features...")
    X_val, y_val = model.prepare_sequences(val_data, apply_sampling=True, feature_set=args.feature_set)
    
    print(f"\nPreparing test sequences with '{args.feature_set}' features...")
    X_test, y_test = model.prepare_sequences(test_data, apply_sampling=False, feature_set=args.feature_set)
=======
    print(f"\nPreparing training sequences with detrended pressure features...")
    X_train, y_train = model.prepare_sequences(train_data, apply_sampling=True, 
                                              use_artifacts=args.use_artifacts, artifact_ratio=args.artifact_ratio)
    
    print(f"\nPreparing validation sequences with detrended pressure features...")
    X_val, y_val = model.prepare_sequences(val_data, apply_sampling=False)
    
    print(f"\nPreparing test sequences with detrended pressure features...")
    X_test, y_test = model.prepare_sequences(test_data, apply_sampling=False)
>>>>>>> origin/2Stage_LSTM
    
    # Print class distribution
    print("\nClass distribution in sets:")
    print(f"Training - Vortex: {sum(y_train)}, Non-vortex: {len(y_train) - sum(y_train)}")
    print(f"Validation - Vortex: {sum(y_val)}, Non-vortex: {len(y_val) - sum(y_val)}")
    print(f"Test - Vortex: {sum(y_test)}, Non-vortex: {len(y_test) - sum(y_test)}")
    
    # Model path
    model_path = Path(__file__).parent.parent / 'models' / args.model_name
    
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
<<<<<<< HEAD
        history = model.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=30, batch_size=128)
=======
        history = model.train(X_train, y_train, X_val, y_val, X_test, y_test, epochs=30, batch_size=args.batch_size, learning_rate=args.learning_rate)
>>>>>>> origin/2Stage_LSTM
        
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
<<<<<<< HEAD
    model.analyze_learned_patterns(test_data, test_results['y_pred'])
=======
    # model.analyze_learned_patterns(test_data, test_results['y_pred'])
>>>>>>> origin/2Stage_LSTM
    
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
<<<<<<< HEAD
        full_results = evaluate_on_full_dataset(model, data, args.feature_set)
=======
        full_results = evaluate_on_full_dataset(model, data)
>>>>>>> origin/2Stage_LSTM
        
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
<<<<<<< HEAD
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
=======
        plot_training_history(history)
>>>>>>> origin/2Stage_LSTM

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

<<<<<<< HEAD
=======
    # At the end, after all evaluation and visualization:
    # Note: Post-processing filters and complex analysis removed for simplified approach

>>>>>>> origin/2Stage_LSTM
if __name__ == "__main__":
    main() 
