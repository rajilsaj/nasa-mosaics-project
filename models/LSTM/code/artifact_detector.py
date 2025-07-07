"""
Artifact Detection and Training Data Enhancement

This module identifies pressure patterns that look like vortex events but are actually
false positives (artifacts) and adds them to the training set as negative examples.
This helps the model learn to distinguish real vortices from artifacts.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArtifactDetector:
    """Detects artifact windows that look like vortex events but are false positives."""
    
    def __init__(self, window_size: int = 60, lookahead: int = 5):
        """
        Initialize the artifact detector.
        
        Args:
            window_size: Size of the pressure window to analyze
            lookahead: Number of points after sharpest drop to analyze for continued drop
        """
        self.window_size = window_size
        self.lookahead = lookahead
        
        # Default artifact detection thresholds (from post-processing filters)
        self.thresholds = {
            'sharp_drop': -0.35,  # Minimum slope threshold for sharp drops
            'total_drop_after_initial': 0.1,  # Minimum total drop after initial drop
            'avg_slope_after_sharpest': -0.1,  # Maximum average slope after sharpest drop (leveling out)
            'total_drop_after_sharpest': -0.75,  # Maximum total drop after sharpest drop
            # Normalized thresholds (based on statistical analysis)
            'normalized_median_pressure': 1.0,  # Artifacts have higher normalized median pressure
            'normalized_pressure_std': 0.3,  # Artifacts have higher normalized pressure variance
            'normalized_slope_std': 0.8  # Artifacts have higher normalized slope variance
        }
    
    def detect_artifacts(self, data: pd.DataFrame, step_size: int = 50) -> List[Dict[str, Any]]:
        """
        Detect artifact windows in the data using sliding window approach.
        
        Args:
            data: DataFrame with 'PRESSURE' column
            step_size: Step size for sliding window (default 50 for efficiency)
            
        Returns:
            List of artifact window information dictionaries
        """
        logger.info(f"Detecting artifacts with window_size={self.window_size}, step_size={step_size}")
        
        artifacts = []
        pressure_values = data['PRESSURE'].values
        
        # Process data with sliding window
        for i in range(self.window_size, len(data), step_size):
            # Get pressure window
            pressure_window = pressure_values[i-self.window_size:i].copy()
            
            # Apply artifact detection criteria
            artifact_info = self._analyze_window_for_artifacts(pressure_window, i)
            
            if artifact_info['is_artifact']:
                artifacts.append(artifact_info)
        
        logger.info(f"Detected {len(artifacts)} artifact windows")
        return artifacts
    
    def _analyze_window_for_artifacts(self, pressure_window: np.ndarray, window_end_idx: int) -> Dict[str, Any]:
        """
        Analyze a single pressure window for artifact characteristics.
        
        Args:
            pressure_window: Pressure values in the window
            window_end_idx: Index where the window ends
            
        Returns:
            Dictionary with artifact analysis results
        """
        # Detrend the pressure window (same as in LSTM model)
        local_mean = np.mean(pressure_window)
        detrended_pressure = pressure_window - local_mean
        
        # Calculate pressure differences
        diffs = np.diff(detrended_pressure)
        
        # Find sharpest drop
        sharp_idx = np.argmin(diffs)
        min_slope = np.min(diffs)
        
        # Calculate artifact features
        features = self._calculate_artifact_features(detrended_pressure, diffs, sharp_idx)
        
        # Determine if this is an artifact based on thresholds
        is_artifact = self._is_artifact_pattern(features)
        
        return {
            'window_end_idx': window_end_idx,
            'pressure_window': pressure_window,
            'detrended_pressure': detrended_pressure,
            'is_artifact': is_artifact,
            'features': features,
            'sharp_idx': sharp_idx,
            'min_slope': min_slope
        }
    
    def _calculate_artifact_features(self, detrended_pressure: np.ndarray, diffs: np.ndarray, sharp_idx: int) -> Dict[str, float]:
        """Calculate features used for artifact detection."""
        features = {}
        
        # 1. Sharp drop threshold
        features['min_slope'] = np.min(diffs)
        
        # 2. Total drop after initial drop
        if np.any(diffs < 0):
            initial_drop_idx = np.argmax(diffs < 0)
            features['total_drop_after_initial'] = detrended_pressure[-1] - detrended_pressure[initial_drop_idx]
        else:
            features['total_drop_after_initial'] = 0
        
        # 3. Average slope after sharpest drop
        after = detrended_pressure[sharp_idx+1:sharp_idx+1+self.lookahead]
        before = detrended_pressure[sharp_idx]
        if len(after) > 0:
            features['avg_slope_after_sharpest'] = (after[-1] - before) / len(after)
        else:
            features['avg_slope_after_sharpest'] = 0
        
        # 4. Total drop after sharpest drop
        features['total_drop_after_sharpest'] = detrended_pressure[-1] - detrended_pressure[sharp_idx]
        
        # 5. Number of consecutive negative slopes after sharpest drop
        neg_count = 0
        for d in diffs[sharp_idx+1:]:
            if d < 0:
                neg_count += 1
            else:
                break
        features['consecutive_negative_slopes'] = neg_count
        
        # 6. Total drop over entire window
        features['total_drop'] = detrended_pressure[-1] - detrended_pressure[0]
        
        # 7. NORMALIZED pressure features (relative to window baseline)
        window_mean = np.mean(detrended_pressure)
        window_std = np.std(detrended_pressure)
        
        # Normalized median pressure (relative to window mean)
        features['normalized_median_pressure'] = np.median(detrended_pressure) - window_mean
        
        # Normalized pressure variance (relative to window)
        features['normalized_pressure_std'] = window_std
        
        # Normalized slope consistency (relative to window)
        features['normalized_slope_std'] = np.std(diffs) / (window_std + 1e-8)
        
        return features
    
    def _is_artifact_pattern(self, features: Dict[str, float]) -> bool:
        """
        Determine if a pattern is an artifact based on thresholds.
        
        Returns True if the pattern meets artifact criteria (should be filtered out).
        """
        # Artifacts are patterns that look like vortices but are actually false positives
        # We want to identify patterns that:
        # 1. Have a sharp drop but don't continue dropping (avg_slope_after_sharpest > -0.3)
        # 2. Have too small a total drop (total_drop_after_initial < 0.1)
        # 3. Drop too much (total_drop_after_sharpest < -0.75)
        
        # Check if it's a sharp drop that doesn't continue dropping
        # For artifacts: sharp drop followed by leveling out (slope close to 0 or positive)
        if (features['min_slope'] < self.thresholds['sharp_drop'] and 
            features['avg_slope_after_sharpest'] > self.thresholds['avg_slope_after_sharpest']):
            return True  # Sharp drop that levels out = artifact
            
        # Check if it's a small drop (not significant enough)
        if features['total_drop_after_initial'] < self.thresholds['total_drop_after_initial']:
            return True  # Drop too small = artifact
            
        # Check if it drops too much (unrealistic)
        if features['total_drop_after_sharpest'] < self.thresholds['total_drop_after_sharpest']:
            return True  # Drops too much = artifact
            
        return False
    
    def prepare_artifact_training_data(self, data: pd.DataFrame, 
                                     artifact_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare artifact windows as negative training examples.
        
        Args:
            data: Input data DataFrame
            artifact_ratio: Ratio of artifacts to include relative to vortex events
            
        Returns:
            Tuple of (sequences, labels) where labels are all 0 (negative)
        """
        logger.info("Preparing artifact training data...")
        
        # Detect artifacts
        artifacts = self.detect_artifacts(data)
        
        if not artifacts:
            logger.warning("No artifacts detected. Returning empty arrays.")
            return np.array([]), np.array([])
        
        # Limit number of artifacts based on ratio
        vortex_count = sum(data['gt_detection_win'] == 1) + sum(data['gt_fwhm'] == 1)
        max_artifacts = int(vortex_count * artifact_ratio)
        
        if len(artifacts) > max_artifacts:
            # Randomly sample artifacts
            selected_indices = np.random.choice(len(artifacts), max_artifacts, replace=False)
            selected_artifacts = [artifacts[i] for i in selected_indices]
        else:
            selected_artifacts = artifacts
        
        # Prepare sequences and labels
        sequences = []
        labels = []
        
        for artifact in selected_artifacts:
            # Create sequence similar to LSTM model (pressure + acceleration)
            detrended_pressure = artifact['detrended_pressure']
            acceleration = np.diff(detrended_pressure, n=2, prepend=[detrended_pressure[0], detrended_pressure[0]])
            sequence = np.stack([detrended_pressure, acceleration], axis=1)
            
            sequences.append(sequence)
            labels.append(0)  # Artifacts are negative examples
        
        logger.info(f"Prepared {len(sequences)} artifact training examples")
        return np.array(sequences), np.array(labels)
    
    def analyze_artifact_distribution(self, artifacts: List[Dict[str, Any]]) -> None:
        """Analyze the distribution of artifact features."""
        if not artifacts:
            logger.warning("No artifacts to analyze")
            return
        
        logger.info("Analyzing artifact distribution...")
        
        # Extract features
        min_slopes = [a['features']['min_slope'] for a in artifacts]
        total_drops_initial = [a['features']['total_drop_after_initial'] for a in artifacts]
        avg_slopes_after = [a['features']['avg_slope_after_sharpest'] for a in artifacts]
        total_drops_sharpest = [a['features']['total_drop_after_sharpest'] for a in artifacts]
        
        logger.info(f"Artifact Statistics:")
        logger.info(f"  Min slope: mean={np.mean(min_slopes):.3f}, std={np.std(min_slopes):.3f}")
        logger.info(f"  Total drop after initial: mean={np.mean(total_drops_initial):.3f}, std={np.std(total_drops_initial):.3f}")
        logger.info(f"  Avg slope after sharpest: mean={np.mean(avg_slopes_after):.3f}, std={np.std(avg_slopes_after):.3f}")
        logger.info(f"  Total drop after sharpest: mean={np.mean(total_drops_sharpest):.3f}, std={np.std(total_drops_sharpest):.3f}")


def integrate_artifacts_with_training_data(train_data: pd.DataFrame, 
                                         val_data: pd.DataFrame,
                                         window_size: int = 60,
                                         artifact_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate artifact windows with existing training data preparation.
    
    Args:
        train_data: Training data DataFrame
        val_data: Validation data DataFrame  
        window_size: Window size for analysis
        artifact_ratio: Ratio of artifacts to include relative to vortex events
        
    Returns:
        Tuple of (X_train_with_artifacts, y_train_with_artifacts, X_val_with_artifacts, y_val_with_artifacts)
    """
    logger.info("Integrating artifacts with training data...")
    
    # Initialize artifact detector
    detector = ArtifactDetector(window_size=window_size)
    
    # Prepare artifact training data
    X_train_artifacts, y_train_artifacts = detector.prepare_artifact_training_data(train_data, artifact_ratio)
    X_val_artifacts, y_val_artifacts = detector.prepare_artifact_training_data(val_data, artifact_ratio)
    
    logger.info(f"Generated {len(X_train_artifacts)} training artifacts and {len(X_val_artifacts)} validation artifacts")
    
    return X_train_artifacts, y_train_artifacts, X_val_artifacts, y_val_artifacts


if __name__ == "__main__":
    # Test the artifact detector
    import sys
    from pathlib import Path
    
    # Add parent directory to path for imports
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    
    # Load test data
    data_path = Path(__file__).parent.parent.parent.parent / 'data' / 'ml_ready_vortex_data.csv'
    data = pd.read_csv(data_path)
    
    # Test artifact detection
    detector = ArtifactDetector(window_size=60)
    artifacts = detector.detect_artifacts(data.iloc[:10000])  # Test on first 10k samples
    
    print(f"Detected {len(artifacts)} artifacts in test data")
    if artifacts:
        detector.analyze_artifact_distribution(artifacts) 