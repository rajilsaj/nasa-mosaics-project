import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import Tuple, Optional

class FeatureProcessor:
    # Define required columns as class constants
    REQUIRED_COLUMNS = [
        'PRESSURE', 'PRESSURE_MA_500',  # Base pressure columns
        'gt_detection_win'  # Ground truth column
    ]
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.features_path = self.data_dir / 'preprocessed_features.joblib'
        self.features: Optional[np.ndarray] = None
        self.full_data: Optional[pd.DataFrame] = None
        
    def validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data structure and content."""
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check data types
        if not pd.api.types.is_numeric_dtype(data['PRESSURE']):
            raise TypeError("PRESSURE column must be numeric")
            
        # Check for missing values
        if data[self.REQUIRED_COLUMNS].isnull().any().any():
            raise ValueError("Data contains missing values")
            
        # Check minimum data length for window calculations
        window_size = 50  # Window size for feature calculation
        min_samples = window_size + 1  # Need at least window_size + 1 samples
        if len(data) < min_samples:
            raise ValueError(f"Data must have at least {min_samples} samples (window_size + 1)")
    
    def calculate_features(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate the 10 features using vectorized operations."""
        # Calculate pressure difference first
        pressure_diff = data['PRESSURE'] - data['PRESSURE_MA_500']
        
        # Create sliding window features using numpy's stride_tricks
        window_size = 50
        n_samples = len(pressure_diff) - window_size + 1
        
        # Create windows using stride_tricks for memory efficiency
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(pressure_diff.values, window_size)
        
        # Calculate all features at once using vectorized operations
        features = np.zeros((n_samples, 10))
        
        # Window-wide features
        features[:, 0] = np.mean(windows, axis=1)  # Mean pressure
        features[:, 1] = np.std(windows, axis=1)   # Pressure variability
        features[:, 2] = np.min(windows, axis=1)   # Minimum pressure
        features[:, 3] = np.max(windows, axis=1)   # Maximum pressure
        
        # Calculate rate of change for all windows at once
        window_diffs = np.diff(windows, axis=1)
        features[:, 4] = np.mean(window_diffs, axis=1)  # Average rate of change
        features[:, 5] = np.std(window_diffs, axis=1)   # Rate of change variability
        
        # Recent features (last 10 samples)
        recent_windows = windows[:, -10:]
        features[:, 6] = np.mean(recent_windows, axis=1)  # Recent pressure mean
        features[:, 7] = np.std(recent_windows, axis=1)   # Recent pressure variability
        
        # Recent rate of change features
        recent_diffs = np.diff(recent_windows, axis=1)
        features[:, 8] = np.mean(recent_diffs, axis=1)  # Recent rate of change
        features[:, 9] = np.std(recent_diffs, axis=1)   # Recent rate of change variability
        
        return features
    
    def _validate_cache_consistency(self, cached_data: np.ndarray, current_data: pd.DataFrame) -> bool:
        """Validate that cached features are consistent with current data."""
        print(f"Cache validation: cached_data length={len(cached_data)}, current_data length={len(current_data)}, window_size=50")
        if len(cached_data) != len(current_data) - 50:  # Account for window size
            return False
        return True
    
    def _get_slice(self, data: pd.DataFrame) -> np.ndarray:
        """Get the appropriate slice of cached features."""
        start_idx = data.index[0]
        end_idx = data.index[-1] - 49  # Account for window size
        return self.features[start_idx:end_idx + 1]
    
    def prepare_features(self, data: pd.DataFrame, force_recalculate: bool = False) -> np.ndarray:
        """Prepare features, using cached version if available and not forced to recalculate."""
        # Validate input data
        self.validate_data(data)
        
        # If we don't have the full data cached, calculate it
        if self.full_data is None or force_recalculate:
            if not force_recalculate and self.features_path.exists():
                print("Loading preprocessed features...")
                cached_data = joblib.load(self.features_path)
                print(f"Loaded cache: shape={cached_data.shape}, current data length={len(data)}")
                
                # Validate cache consistency
                print("Validating cache consistency...")
                if not self._validate_cache_consistency(cached_data, data):
                    print("Cache validation failed, recalculating features...")
                    force_recalculate = True
                else:
                    print("Cache validation passed, using cached features...")
                    self.features = cached_data
                    self.full_data = data
                    print("Returning cached features...")
                    return self._get_slice(data)
            
            if force_recalculate:
                print("Calculating features...")
                self.features = self.calculate_features(data)
                self.full_data = data
                
                # Save the preprocessed features
                print("Saving preprocessed features...")
                joblib.dump(self.features, self.features_path)
        
        return self._get_slice(data)
    
    def clear_cache(self):
        """Clear the cached features."""
        if self.features_path.exists():
            self.features_path.unlink()
        self.features = None
        self.full_data = None 