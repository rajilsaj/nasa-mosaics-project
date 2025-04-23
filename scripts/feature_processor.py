import numpy as np
import pandas as pd

class FeatureProcessor:
    """
    Extracts statistical features from a window of pressure time-series data.

    Attributes:
    -----------
    window_size : int
        Total number of samples in the window to analyze.
    recent_focus_size : int
        Number of most recent samples to focus on for localized analysis.
    """

    def __init__(self, window_size: int = 50, recent_focus_size: int = 10):
        self.window_size = window_size
        self.recent_focus_size = recent_focus_size

    def compute_features(self, df_window: pd.DataFrame) -> dict:
        """
        Compute statistical features from a DataFrame window of pressure data.

        Parameters:
        -----------
        df_window : pd.DataFrame
            A DataFrame containing at least a 'PRESSURE' column with
            a minimum number of rows equal to window_size.

        Returns:
        --------
        dict
            A dictionary of 10 statistical features.
        """
        if len(df_window) < self.window_size or "PRESSURE" not in df_window.columns:
            return {}

        pressure = df_window["PRESSURE"].values
        recent = pressure[-self.recent_focus_size:]

        return {
            "mean_pressure": np.mean(pressure),
            "std_pressure": np.std(pressure),
            "min_pressure": np.min(pressure),
            "max_pressure": np.max(pressure),
            "pressure_change": pressure[-1] - pressure[0],
            "recent_mean": np.mean(recent),
            "recent_std": np.std(recent),
            "recent_min": np.min(recent),
            "recent_max": np.max(recent)
        }
