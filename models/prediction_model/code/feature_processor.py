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
        """Calculate the 10 features we actually use."""
        # Calculate pressure difference first
        pressure_diff = data['PRESSURE'] - data['PRESSURE_MA_500']
        
        # Create sliding window features
        window_size = 50
        features = []
        
        for i in range(window_size, len(data)):
            window = pressure_diff.iloc[i-window_size:i]
            
            # Calculate the 10 features in the same order as FEATURE_NAMES
            features.append([
                np.mean(window[-10:]),     # Recent Pressure Mean
                np.mean(np.diff(window[-10:])),  # Recent Rate of Change
                np.std(window),            # Pressure Variability
                np.mean(np.diff(window)),  # Average Rate of Change
                np.std(window[-10:]),      # Recent Pressure Variability
                np.std(np.diff(window)),   # Rate of Change Variability
                np.std(np.diff(window[-10:])),  # Recent Rate of Change Variability
                np.mean(window),           # Mean Pressure
                np.min(window),            # Min Pressure
                np.max(window)             # Max Pressure
            ])
        
        return np.array(features)
    
    def _validate_cache_consistency(self, cached_data: np.ndarray, 
                                  current_data: pd.DataFrame) -> bool:
        """Validate that cached features are consistent with current data."""
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
                
                # Validate cache consistency
                if not self._validate_cache_consistency(cached_data, data):
                    print("Cache validation failed, recalculating features...")
                    force_recalculate = True
                else:
                    self.features = cached_data
                    self.full_data = data
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