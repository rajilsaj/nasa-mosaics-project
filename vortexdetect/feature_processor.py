import numpy as np
import pandas as pd
from typing import Dict


class FeatureProcessor:
    """
    Extracts statistical features from a window of pressure time-series data.

    Attributes
    ----------
    window_size : int
        Total number of samples to consider in the window.
    recent_focus_size : int
        Number of most recent samples for short-term trend analysis.
    """

    def __init__(self, window_size: int = 50, recent_focus_size: int = 10):
        self.window_size = window_size
        self.recent_focus_size = recent_focus_size

    def compute_features(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """
        Compute statistical features from a time-series pressure window.

        Parameters
        ----------
        df_window : pd.DataFrame
            A DataFrame containing a 'PRESSURE' column with at least `window_size` rows.

        Returns
        -------
        Dict[str, float]
            A dictionary of extracted features including global and recent stats.
        """
        if len(df_window) < self.window_size or "PRESSURE" not in df_window.columns:
            return {}

        pressure = df_window["PRESSURE"].values
        recent = pressure[-self.recent_focus_size:]

        features = {
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

        return features
