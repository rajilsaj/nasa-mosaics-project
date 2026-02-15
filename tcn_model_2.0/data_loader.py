"""
Data loader for TCN model training on bow shock crossing events.

Loads NetCDF files and creates sequences with labels for bow shock events.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import xarray as xr


def parse_yaml_labels(yaml_path: Path) -> dict[str, list[str]] | None:
    """Parse YAML labels file."""
    if not yaml_path.exists():
        return None
    
    data: dict[str, list[str]] = {
        "change_points": [],
        "bimodality": [],
        "negative_ions": [],
    }
    
    with open(yaml_path, "r", encoding="utf-8") as fh:
        current_key: str | None = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":"):
                current_key = line[:-1].strip()
                if current_key not in data:
                    data[current_key] = []
            elif line.startswith("- "):
                value = line[2:].strip()
                if current_key and current_key in data:
                    data[current_key].append(value)
    return data


def get_bow_shock_times(yaml_path: Path) -> list[datetime]:
    """
    Extract bow shock crossing event times from YAML file.
    
    Bow shock is the second change_point (index 1) in the labels.
    If there's only one change_point, it might be bow shock (some files only have bow shock).
    """
    labels = parse_yaml_labels(yaml_path)
    if not labels or "change_points" not in labels:
        return []
    
    change_points = labels["change_points"]
    bow_shock_times = []
    
    # Bow shock is the second change_point (index 1)
    # If there's only one change_point, check if it might be bow shock
    # (Some files might only have bow shock without magnetopause)
    if len(change_points) > 1:
        # Standard case: second change_point is bow shock
        bs_str = change_points[1]
    elif len(change_points) == 1:
        # Some files might only have bow shock (no magnetopause)
        # Use the single change_point as bow shock
        bs_str = change_points[0]
    else:
        return []
    
    try:
        dt = datetime.strptime(bs_str, "%d-%m-%Y/%H:%M:%S")
        bow_shock_times.append(dt)
    except ValueError:
        try:
            dt = datetime.strptime(bs_str, "%d-%m-%Y/%H:%M")
            bow_shock_times.append(dt)
        except ValueError:
            pass
    
    return bow_shock_times


def load_nc_data(nc_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NetCDF file and extract count rate data.
    
    Applies log scaling: log10(counts + 1) to handle log-normal distribution
    of count rates. This is done before normalization.
    
    Args:
        nc_path: Path to NetCDF file
    
    Returns:
        Tuple of (count_rate, times)
        - count_rate: (time, energy) array (averaged over anodes, log-scaled)
        - times: Array of datetime64 timestamps
    
    Raises:
        ImportError: If netcdf4 backend is not installed
        Exception: If file cannot be loaded
    """
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        # Check if it's a missing backend issue
        error_msg = str(e).lower()
        if 'netcdf4' in error_msg or 'h5netcdf' in error_msg or 'backend' in error_msg:
            raise ImportError(
                f"NetCDF backend not installed. Install with: pip install netcdf4\n"
                f"Original error: {e}"
            ) from e
        else:
            raise
    
    # Load count rate data
    counts = ds["count_rate"].values.astype(float)
    counts[counts == 65535.0] = np.nan
    
    # Average over anodes: (time, energy, anode) -> (time, energy)
    # Handle case where anode dimension might be missing or empty
    if counts.ndim == 3 and counts.shape[2] > 0:
        counts = np.nanmean(counts, axis=2)
    elif counts.ndim == 2:
        # Already 2D, no anode dimension to average
        pass
    else:
        # Unexpected shape, try to handle gracefully
        if counts.ndim == 3 and counts.shape[2] == 0:
            # Empty anode dimension - take first slice or create 2D array
            counts = counts[:, :, 0] if counts.shape[0] > 0 and counts.shape[1] > 0 else np.zeros((counts.shape[0], counts.shape[1]))
        else:
            raise ValueError(f"Unexpected count_rate shape: {counts.shape}, expected (time, energy, anode) or (time, energy)")
    
    # Interpolate NaN values along time axis
    for energy_idx in range(counts.shape[1]):
        column = counts[:, energy_idx]
        finite_mask = np.isfinite(column)
        if finite_mask.sum() >= 2:
            missing_idx = np.where(~finite_mask)[0]
            known_idx = np.where(finite_mask)[0]
            known_vals = column[finite_mask]
            if len(missing_idx) > 0:
                column[~finite_mask] = np.interp(missing_idx, known_idx, known_vals)
                counts[:, energy_idx] = column
    
    # Replace any remaining NaN with 0
    counts = np.nan_to_num(counts, nan=0.0)
    
    # Apply log scaling: log10(counts + 1) to handle log-normal distribution
    # +1 prevents log(0) and preserves zero values
    counts = np.log10(counts + 1.0)
    
    times = ds["time"].values
    ds.close()
    
    return counts, times


def create_bow_shock_labels(
    times: np.ndarray,
    yaml_path: Path,
    window_size: int = 5,
) -> np.ndarray:
    """
    Create binary labels for bow shock events.
    
    Args:
        times: Array of datetime64 timestamps
        yaml_path: Path to YAML labels file
        window_size: Number of time steps around event to label as positive
    
    Returns:
        Binary array of shape (time,) with 1s at bow shock events
    """
    labels = np.zeros(len(times), dtype=np.int64)
    
    bow_shock_times = get_bow_shock_times(yaml_path)
    if not bow_shock_times:
        return labels
    
    # Convert times to datetime64 for comparison
    times_dt64 = pd.to_datetime(times).to_numpy()
    
    for bs_time in bow_shock_times:
        bs_dt64 = np.datetime64(bs_time)
        
        # Find closest time index
        time_diffs = np.abs(times_dt64 - bs_dt64)
        closest_idx = np.argmin(time_diffs)
        min_diff = time_diffs[closest_idx]
        
        # Convert timedelta to hours for checking
        min_diff_hours = min_diff.astype('timedelta64[h]').astype(float)
        
        # Only label if the time difference is reasonable (within 24 hours)
        # Increased from 2 hours to handle larger time mismatches
        # This handles cases where YAML times don't exactly match NetCDF times
        if min_diff_hours <= 24.0:
            # Label window_size time steps around the event
            start_idx = max(0, closest_idx - window_size // 2)
            end_idx = min(len(labels), closest_idx + window_size // 2 + 1)
            labels[start_idx:end_idx] = 1
        else:
            # Debug: log when time difference is too large (but only for first few events to avoid spam)
            pass  # Debugging moved to dataset level
    
    return labels


class BowShockDataset(Dataset):
    """
    Dataset for bow shock crossing event detection.
    
    Creates sliding windows of sequence data with labels.
    Can work in two modes:
    1. YAML-based: Uses YAML files to mark specific time points (window_size around events)
    2. Directory-based: All sequences from a directory get the same label (all 1s or all 0s)
    """
    
    def __init__(
        self,
        nc_files: list[Path],
        labels_dir: Optional[Path] = None,
        sequence_length: int = 100,
        stride: int = 10,
        normalize: bool = True,
        window_size: int = 5,
            normalize_mean: Optional[np.ndarray] = None,
            normalize_std: Optional[np.ndarray] = None,
        directory_label: Optional[int] = None,
        max_samples_per_file: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            nc_files: List of NetCDF file paths
            labels_dir: Directory containing YAML label files (optional, for YAML-based mode)
            sequence_length: Length of input sequences
            stride: Stride for sliding window
            normalize: Whether to normalize features (per-energy-bin normalization)
            window_size: Number of time steps around event to label as positive (YAML mode)
            normalize_mean: Mean array for per-energy-bin normalization, shape (num_energy_bins,)
                           (if None, compute from data)
            normalize_std: Std array for per-energy-bin normalization, shape (num_energy_bins,)
                          (if None, compute from data)
            directory_label: If provided, all sequences get this label (1 for true, 0 for false)
                            If None, uses YAML-based labeling
            max_samples_per_file: Maximum number of sequences to create per file (for faster testing)
        """
        self.nc_files = nc_files
        self.labels_dir = labels_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.window_size = window_size
        self.directory_label = directory_label
        self.max_samples_per_file = max_samples_per_file
        
        # Load all data
        self.sequences = []
        self.labels = []
        self.file_indices = []
        
        print(f"Loading {len(nc_files)} files...")
        
        # Determine labeling mode
        if directory_label is not None:
            print(f"Using directory-based labeling: all sequences = {directory_label}")
            use_yaml_labels = False
        else:
            print(f"Using YAML-based labeling")
            if labels_dir:
                print(f"Labels directory: {labels_dir}")
                print(f"Labels directory exists: {labels_dir.exists()}")
            use_yaml_labels = True
        
        # Track statistics
        files_with_yaml = 0
        files_with_events = 0
        total_events_found = 0
        
        for file_idx, nc_path in enumerate(nc_files):
            try:
                # Load data
                counts, times = load_nc_data(nc_path)
                
                # Get labels based on mode
                if use_yaml_labels and labels_dir:
                    # YAML-based: use YAML files to mark specific time points
                    yaml_path = labels_dir / f"{nc_path.stem}.yaml"
                    if yaml_path.exists():
                        files_with_yaml += 1
                    
                    if not yaml_path.exists() and file_idx == 0:
                        print(f"  Warning: Label file not found for {nc_path.name}")
                        print(f"  Expected: {yaml_path}")
                        # List some available YAML files for debugging
                        available_yamls = list(labels_dir.glob("*.yaml"))[:5]
                        if available_yamls:
                            print(f"  Available YAML files (first 5): {[f.name for f in available_yamls]}")
                    
                    file_labels = create_bow_shock_labels(times, yaml_path, window_size)
                    
                    # Track events
                    positive_in_file = np.sum(file_labels == 1)
                    if positive_in_file > 0:
                        files_with_events += 1
                        # Count number of events (approximate by counting label transitions)
                        label_changes = np.diff(np.concatenate([[0], file_labels, [0]]))
                        total_events_found += np.sum(label_changes == 1)
                    
                    # Debug: Check if labels were created (for ALL files with events but no positives)
                    positive_count = np.sum(file_labels == 1)
                    if positive_count == 0 and yaml_path.exists():
                        bs_times = get_bow_shock_times(yaml_path)
                        if bs_times:  # Only warn if YAML actually has events
                            print(f"  ⚠ Warning: No positive labels for {nc_path.name}")
                            print(f"    YAML file: {yaml_path.name} (exists: {yaml_path.exists()})")
                            print(f"    Bow shock times from YAML: {bs_times}")
                            if len(times) > 0:
                                times_dt = pd.to_datetime(times)
                                print(f"    NetCDF time range: {times_dt[0]} to {times_dt[-1]}")
                                print(f"    NetCDF time points: {len(times)}")
                                
                                # Check time differences
                                times_dt64 = times_dt.to_numpy()
                                for bs_time in bs_times:
                                    bs_dt64 = np.datetime64(bs_time)
                                    time_diffs = np.abs(times_dt64 - bs_dt64)
                                    min_diff = time_diffs.min()
                                    min_diff_hours = min_diff.astype('timedelta64[h]').astype(float)
                                    closest_idx = np.argmin(time_diffs)
                                    print(f"    Closest match to {bs_time}:")
                                    print(f"      Index: {closest_idx}, Time: {times_dt[closest_idx]}")
                                    print(f"      Time difference: {min_diff_hours:.2f} hours")
                                    if min_diff_hours > 24.0:
                                        print(f"      ⚠ Too large (>24 hours) - won't be labeled!")
                                    else:
                                        print(f"      ✓ Within 24 hours - should be labeled!")
                                        print(f"      Window size: {window_size}, would label indices {max(0, closest_idx - window_size // 2)} to {min(len(times), closest_idx + window_size // 2 + 1)}")
                            else:
                                print(f"    ⚠ No time data in NetCDF file!")
                    elif not yaml_path.exists() and file_idx == 0:
                        print(f"  Warning: Label file not found for {nc_path.name}")
                        print(f"  Expected: {yaml_path}")
                        # List some available YAML files for debugging
                        available_yamls = list(labels_dir.glob("*.yaml"))[:5]
                        if available_yamls:
                            print(f"  Available YAML files (first 5): {[f.name for f in available_yamls]}")
                else:
                    # Directory-based: all sequences get the same label
                    file_labels = np.full(len(counts), directory_label, dtype=np.int64)
                
                # Create sliding windows (limit samples per file for faster testing)
                sequences_from_file = []
                labels_from_file = []
                
                for start_idx in range(0, len(counts) - sequence_length + 1, stride):
                    end_idx = start_idx + sequence_length
                    sequence = counts[start_idx:end_idx]
                    label = file_labels[start_idx:end_idx]
                    
                    sequences_from_file.append(sequence)
                    labels_from_file.append(label)
                    
                    # Limit samples per file if specified
                    if max_samples_per_file and len(sequences_from_file) >= max_samples_per_file:
                        break
                
                self.sequences.extend(sequences_from_file)
                self.labels.extend(labels_from_file)
                self.file_indices.extend([file_idx] * len(sequences_from_file))
                
                if (file_idx + 1) % 10 == 0:
                    print(f"  Processed {file_idx + 1}/{len(nc_files)} files...")
            
            except ImportError as e:
                # Missing dependency - raise immediately
                print(f"\n❌ CRITICAL ERROR: {e}")
                print(f"   File: {nc_path.name}")
                print(f"\n   SOLUTION: Install the missing dependency:")
                print(f"   pip install netcdf4")
                raise
            except Exception as e:
                # Other errors - log and continue
                error_msg = str(e)
                # Only print first few errors to avoid spam
                if file_idx < 5:
                    print(f"  Warning: Failed to load {nc_path.name}: {error_msg[:100]}")
                elif file_idx == 5:
                    print(f"  ... (suppressing further load errors)")
                continue
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        # Normalize features (per-energy-bin normalization)
        if normalize:
            if normalize_mean is not None and normalize_std is not None:
                # Use provided normalization parameters
                # normalize_mean and normalize_std should be arrays of shape (num_energy_bins,)
                self.mean = normalize_mean
                self.std = normalize_std
            else:
                # Compute mean and std per energy bin across all sequences
                # sequences shape: (n_sequences, sequence_length, num_energy_bins)
                # Reshape to (n_sequences * sequence_length, num_energy_bins)
                n_sequences, seq_len, num_energy_bins = self.sequences.shape
                sequences_flat = self.sequences.reshape(-1, num_energy_bins)
                
                # Compute mean and std for each energy bin
                self.mean = np.nanmean(sequences_flat, axis=0)  # Shape: (num_energy_bins,)
                self.std = np.nanstd(sequences_flat, axis=0)  # Shape: (num_energy_bins,)
                
                # Avoid division by zero
                self.std = np.where(self.std < 1e-8, 1.0, self.std)
            
            # Normalize each energy bin independently
            # self.mean and self.std are shape (num_energy_bins,)
            # Broadcasting will apply normalization per energy bin
            self.sequences = (self.sequences - self.mean) / self.std
            # Replace any NaN with 0
            self.sequences = np.nan_to_num(self.sequences, nan=0.0)
        else:
            self.mean = None
            self.std = None
        
        print(f"Created {len(self.sequences)} sequences from {len(nc_files)} files")
        positive_samples = np.sum(self.labels == 1)
        negative_samples = np.sum(self.labels == 0)
        print(f"  Positive samples: {positive_samples}")
        print(f"  Negative samples: {negative_samples}")
        
        if use_yaml_labels and labels_dir:
            print(f"\n  YAML Label Statistics:")
            print(f"    Files with matching YAML: {files_with_yaml}/{len(nc_files)}")
            print(f"    Files with detected events: {files_with_events}")
            print(f"    Total events found: {total_events_found}")
        
        if positive_samples == 0 and use_yaml_labels:
            print("\n  ⚠ WARNING: No positive samples found!")
            print("  This means no bow shock events were detected in the labels.")
            print("  Possible causes:")
            print("    1. YAML files don't exist or aren't being found")
            print("    2. YAML files don't have at least 2 change_points")
            print("    3. Time matching between NetCDF and YAML is failing (time difference > 24 hours)")
            print("    4. Date format in YAML doesn't match expected format")
            print("    5. NetCDF time range doesn't overlap with YAML event times")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence and its labels.
        
        Returns:
            Tuple of (sequence, labels)
            - sequence: (sequence_length, num_energy_bins)
            - labels: (sequence_length,)
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        labels = torch.LongTensor(self.labels[idx])
        
        return sequence, labels


def create_data_loaders(
    true_processed_dir: Path,
    false_processed_dir: Path,
    sequence_length: int = 100,
    stride: int = 10,
    train_split: float = 0.7,
    val_split: float = 0.15,
    batch_size: int = 32,
    normalize: bool = True,
    num_workers: int = 0,
    normalize_mean: Optional[np.ndarray] = None,
    normalize_std: Optional[np.ndarray] = None,
    max_files: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Directory-based labeling approach:
    - true_processed: All sequences are labeled as positive (1) - contains bow shock events
    - false_processed: All sequences are labeled as negative (0) - no bow shock events
    
    Args:
        true_processed_dir: Directory containing true processed NetCDF files (with bow shock events)
        false_processed_dir: Directory containing false processed NetCDF files (no bow shock events)
        sequence_length: Length of input sequences (default: 100)
        stride: Stride for sliding window (default: 10)
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        batch_size: Batch size (default: 32)
        normalize: Whether to normalize features (per-energy-bin normalization)
        num_workers: Number of worker processes for data loading
        normalize_mean: Mean array for per-energy-bin normalization, shape (num_energy_bins,)
                       (if None, compute from training data)
        normalize_std: Std array for per-energy-bin normalization, shape (num_energy_bins,)
                      (if None, compute from training data)
        max_files: Maximum number of files to use from each directory (default: None, uses all files)
        max_samples_per_file: Maximum sequences per file (default: None, uses all sequences)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Find all NetCDF files from both directories
    true_nc_files = sorted(true_processed_dir.rglob("*.nc"))
    false_nc_files = sorted(false_processed_dir.rglob("*.nc"))
    
    print(f"Found {len(true_nc_files)} true_processed files")
    print(f"Found {len(false_nc_files)} false_processed files")
    
    # Limit files if max_files is specified
    if max_files is not None:
        if len(true_nc_files) > max_files:
            print(f"⚠ Limiting true_processed files from {len(true_nc_files)} to {max_files}")
            true_nc_files = true_nc_files[:max_files]
        if len(false_nc_files) > max_files:
            print(f"⚠ Limiting false_processed files from {len(false_nc_files)} to {max_files}")
            false_nc_files = false_nc_files[:max_files]
    
    if len(true_nc_files) == 0:
        raise ValueError(
            f"No NetCDF files found in {true_processed_dir}\n"
            f"Directory exists: {true_processed_dir.exists()}\n"
            f"Directory is directory: {true_processed_dir.is_dir() if true_processed_dir.exists() else 'N/A'}"
        )
    
    if len(false_nc_files) == 0:
        raise ValueError(
            f"No NetCDF files found in {false_processed_dir}\n"
            f"Directory exists: {false_processed_dir.exists()}\n"
            f"Directory is directory: {false_processed_dir.is_dir() if false_processed_dir.exists() else 'N/A'}"
        )
    
    print(f"Found {len(true_nc_files)} true_processed files (all labeled as positive/events)")
    print(f"Found {len(false_nc_files)} false_processed files (all labeled as negative/no events)")
    print("Using directory-based labeling (no YAML files required)")
    print()
    
    # Split true files into train/val/test
    n_true_files = len(true_nc_files)
    n_train_true = int(n_true_files * train_split)
    n_val_true = int(n_true_files * val_split)
    
    train_true_files = true_nc_files[:n_train_true]
    val_true_files = true_nc_files[n_train_true:n_train_true + n_val_true]
    test_true_files = true_nc_files[n_train_true + n_val_true:]
    
    # Split false files into train/val/test
    n_false_files = len(false_nc_files)
    n_train_false = int(n_false_files * train_split)
    n_val_false = int(n_false_files * val_split)
    
    train_false_files = false_nc_files[:n_train_false]
    val_false_files = false_nc_files[n_train_false:n_train_false + n_val_false]
    test_false_files = false_nc_files[n_train_false + n_val_false:]
    
    print(f"File split:")
    print(f"  True files (all positive): {len(train_true_files)} train, {len(val_true_files)} val, {len(test_true_files)} test")
    print(f"  False files (all negative): {len(train_false_files)} train, {len(val_false_files)} val, {len(test_false_files)} test")
    print()
    
    # Compute normalization stats from all training files (both true and false)
    # Per-energy-bin normalization: compute mean and std for each energy bin independently
    if normalize and normalize_mean is None:
        print("Computing per-energy-bin normalization statistics from training data...")
        all_counts = []
        sample_files = (train_true_files + train_false_files)[:min(20, len(train_true_files) + len(train_false_files))]
        for nc_path in sample_files:
            try:
                counts, _ = load_nc_data(nc_path)
                all_counts.append(counts)
            except ImportError as e:
                # Missing dependency - raise immediately
                print(f"\n❌ CRITICAL ERROR: {e}")
                print(f"   File: {nc_path.name}")
                print(f"\n   SOLUTION: Install the missing dependency:")
                print(f"   pip install netcdf4")
                raise
            except Exception as e:
                # Other errors - skip this file
                print(f"  Warning: Failed to load {nc_path.name}: {str(e)[:100]}")
                continue
        if all_counts:
            all_counts = np.concatenate(all_counts, axis=0)  # Shape: (total_time, num_energy_bins)
            # Compute mean and std for each energy bin independently
            normalize_mean = np.nanmean(all_counts, axis=0)  # Shape: (num_energy_bins,)
            normalize_std = np.nanstd(all_counts, axis=0)  # Shape: (num_energy_bins,)
            # Avoid division by zero
            normalize_std = np.where(normalize_std < 1e-8, 1.0, normalize_std)
            print(f"  Per-energy-bin normalization:")
            print(f"    Mean range: [{normalize_mean.min():.4f}, {normalize_mean.max():.4f}]")
            print(f"    Std range: [{normalize_std.min():.4f}, {normalize_std.max():.4f}]")
            print(f"    Number of energy bins: {len(normalize_mean)}")
        else:
            # Fallback: if no files loaded, create dummy normalization
            # This shouldn't happen, but handle gracefully
            print("  Warning: No files loaded for normalization, using defaults")
            # We'll need to get num_energy_bins from somewhere - use a default
            # This will be corrected when the first dataset is created
            normalize_mean = None
            normalize_std = None
        print()
    
    val_norm_mean = normalize_mean if normalize else None
    val_norm_std = normalize_std if normalize else None
    
    # Create datasets
    # True files: All sequences are positive (directory_label=1)
    # False files: All sequences are negative (directory_label=0)
    from torch.utils.data import ConcatDataset
    
    print("Creating training datasets...")
    train_datasets = []
    
    # True processed: All sequences are positive
    if len(train_true_files) > 0:
        train_true_dataset = BowShockDataset(
            train_true_files,
            labels_dir=None,  # No YAML labels needed
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            window_size=5,  # Not used for directory-based labeling, but kept for compatibility
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            directory_label=1,  # All positive
            max_samples_per_file=max_samples_per_file,
        )
        train_datasets.append(train_true_dataset)
    
    # False processed: All sequences are negative
    if len(train_false_files) > 0:
        train_false_dataset = BowShockDataset(
            train_false_files,
            labels_dir=None,
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            window_size=5,  # Not used for directory-based labeling, but kept for compatibility
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            directory_label=0,  # All negative
            max_samples_per_file=max_samples_per_file,
        )
        train_datasets.append(train_false_dataset)
    
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    
    print("Creating validation datasets...")
    val_datasets = []
    
    if len(val_true_files) > 0:
        val_true_dataset = BowShockDataset(
            val_true_files,
            labels_dir=None,  # No YAML labels needed
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            window_size=5,  # Not used for directory-based labeling
            normalize_mean=val_norm_mean,
            normalize_std=val_norm_std,
            directory_label=1,  # All positive
            max_samples_per_file=max_samples_per_file,
        )
        val_datasets.append(val_true_dataset)
    
    if len(val_false_files) > 0:
        val_false_dataset = BowShockDataset(
            val_false_files,
            labels_dir=None,
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            window_size=5,  # Not used for directory-based labeling
            normalize_mean=val_norm_mean,
            normalize_std=val_norm_std,
            directory_label=0,  # All negative
            max_samples_per_file=max_samples_per_file,
        )
        val_datasets.append(val_false_dataset)
    
    val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    
    print("Creating test datasets...")
    test_datasets = []
    
    if len(test_true_files) > 0:
        test_true_dataset = BowShockDataset(
            test_true_files,
            labels_dir=None,  # No YAML labels needed
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            window_size=5,  # Not used for directory-based labeling
            normalize_mean=val_norm_mean,
            normalize_std=val_norm_std,
            directory_label=1,  # All positive
            max_samples_per_file=max_samples_per_file,
        )
        test_datasets.append(test_true_dataset)
    
    if len(test_false_files) > 0:
        test_false_dataset = BowShockDataset(
            test_false_files,
            labels_dir=None,
            sequence_length=sequence_length,
            stride=stride,
            normalize=normalize,
            window_size=5,  # Not used for directory-based labeling
            normalize_mean=val_norm_mean,
            normalize_std=val_norm_std,
            directory_label=0,  # All negative
            max_samples_per_file=max_samples_per_file,
        )
        test_datasets.append(test_false_dataset)
    
    test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]
    print()
    
    # Validate datasets are not empty
    if len(train_dataset) == 0:
        raise ValueError(
            "Training dataset is empty! No sequences were created.\n"
            "This usually means:\n"
            "1. NetCDF files failed to load (check if netcdf4 is installed: pip install netcdf4)\n"
            "2. All files were too short to create sequences with the given sequence_length\n"
            "3. All files failed to load due to errors\n"
            f"Attempted to load {len(train_true_files)} true files and {len(train_false_files)} false files"
        )
    
    if len(val_dataset) == 0:
        raise ValueError(
            "Validation dataset is empty! No sequences were created.\n"
            f"Attempted to load {len(val_true_files)} true files and {len(val_false_files)} false files"
        )
    
    if len(test_dataset) == 0:
        raise ValueError(
            "Test dataset is empty! No sequences were created.\n"
            f"Attempted to load {len(test_true_files)} true files and {len(test_false_files)} false files"
        )
    
    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    return train_loader, val_loader, test_loader
