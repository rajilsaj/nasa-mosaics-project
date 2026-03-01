"""
Data loader for TCN model training on bow shock crossing events.

Labels come from the crossing_event variable inside each .nc file:
  0 → no crossing event
  1 → crossing event (bow shock or magnetopause)

All .nc files live in a single processed directory (new_processed/).
The train/val/test split is done by day-of-year extracted from the filename,
matching the roadmap split: train days 1-270, val 271-335, test 336-366.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple
import xarray as xr
import re


# ---------------------------------------------------------------------------
# NetCDF loading
# ---------------------------------------------------------------------------

def load_nc_data(nc_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a NetCDF file and return log-scaled count-rate data and labels.

    Steps:
      1. Load count_rate (time, energy, anode)
      2. Replace sentinel 65535 values with NaN
      3. Average across the anode dimension → (time, energy)
      4. Clip negatives to 0, then interpolate remaining NaNs along time axis
      5. Apply log10(x + 1e-6) to compress the dynamic range
      6. Load the crossing_event variable as the per-timestep label array

    Returns:
        counts  : np.ndarray, shape (time, energy)  — log-scaled energy data
        labels  : np.ndarray, shape (time,)          — 0/1 per timestep
        times   : np.ndarray of datetime64 timestamps
    """
    ds = xr.open_dataset(nc_path)

    # --- Count rate -------------------------------------------------------
    counts = ds["count_rate"].values.astype(float)
    counts[counts == 65535.0] = np.nan

    # Average over anode dimension if present
    if counts.ndim == 3:
        counts = np.nanmean(counts, axis=2)     # (time, energy)

    # Clip negatives — instrument artefact, not physically meaningful
    counts = np.clip(counts, 0.0, None)

    # Interpolate NaN along time for each energy channel
    for ch in range(counts.shape[1]):
        col = counts[:, ch]
        finite = np.isfinite(col)
        if finite.sum() >= 2 and (~finite).any():
            counts[~finite, ch] = np.interp(
                np.where(~finite)[0],
                np.where(finite)[0],
                col[finite],
            )

    counts = np.nan_to_num(counts, nan=0.0)
    counts = np.log10(counts + 1e-6)           # log-scale (roadmap Step 4a)

    # --- Labels -----------------------------------------------------------
    if "crossing_event" not in ds:
        raise ValueError(
            f"{nc_path.name} has no 'crossing_event' variable. "
            "Run add_crossing_event.py first."
        )
    labels = ds["crossing_event"].values.astype(np.int64)

    times = ds["time"].values
    ds.close()
    return counts, labels, times


# ---------------------------------------------------------------------------
# Day-of-year extraction from filename
# ---------------------------------------------------------------------------

def day_of_year_from_path(nc_path: Path) -> int:
    """
    Extract the day-of-year integer from an ELS filename.

    ELS filenames follow the pattern:  ELS_<YYYYDDD>_<version>_raw.nc
    e.g. ELS_200400100_V01_raw.nc → day 1 of 2004
         ELS_200400365_V01_raw.nc → day 365 of 2004

    The middle three digits of the 9-digit block are the day of year.
    Returns -1 if the pattern is not found (file will be excluded from splits).
    """
    match = re.search(r"ELS_\d{4}(\d{3})\d{2}_", nc_path.name)
    if match:
        return int(match.group(1))
    return -1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BowShockDataset(Dataset):
    """
    Sliding-window dataset over NetCDF files.

    Each window of length `sequence_length` gets a single label derived
    from its centre timestep's crossing_event value. This is cleaner than
    majority voting for change-point detection (roadmap Step 5).

    Args:
        nc_files        : list of NetCDF file paths to load
        sequence_length : number of time steps per window
        stride          : how far to slide the window each step
        normalize_mean  : per-energy-bin mean, shape (num_energy_bins,)
        normalize_std   : per-energy-bin std,  shape (num_energy_bins,)
                          If both None, computed from this dataset's data.
        max_samples_per_file : cap windows per file (useful for quick tests)
    """

    def __init__(
        self,
        nc_files: list[Path],
        sequence_length: int = 128,
        stride: int = 16,
        normalize_mean: Optional[np.ndarray] = None,
        normalize_std: Optional[np.ndarray] = None,
        max_samples_per_file: Optional[int] = None,
    ):
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []

        centre = sequence_length // 2          # index of the centre timestep

        for nc_path in nc_files:
            try:
                counts, file_labels, _ = load_nc_data(nc_path)
            except Exception as e:
                print(f"  Warning: skipping {nc_path.name} — {e}")
                continue

            n = len(counts)
            added = 0
            for start in range(0, n - sequence_length + 1, stride):
                window_label = int(file_labels[start + centre])  # centre timestep label
                self.sequences.append(counts[start : start + sequence_length])
                self.labels.append(window_label)
                added += 1
                if max_samples_per_file and added >= max_samples_per_file:
                    break

        if len(self.sequences) == 0:
            raise ValueError(
                f"No sequences created from {len(nc_files)} files. "
                "Check files load correctly and are longer than sequence_length."
            )

        self.sequences = np.array(self.sequences, dtype=np.float32)  # (N, T, E)
        self.labels    = np.array(self.labels,    dtype=np.int64)     # (N,)

        # Per-channel z-score normalisation (roadmap Step 4b)
        # Fit only on training data — pass mean/std for val and test sets
        if normalize_mean is None or normalize_std is None:
            flat = self.sequences.reshape(-1, self.sequences.shape[2])  # (N*T, E)
            normalize_mean = flat.mean(axis=0)
            normalize_std  = flat.std(axis=0)
            normalize_std  = np.where(normalize_std < 1e-8, 1.0, normalize_std)

        self.mean = normalize_mean
        self.std  = normalize_std
        self.sequences = (self.sequences - self.mean) / self.std
        self.sequences = np.nan_to_num(self.sequences, nan=0.0)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (sequence, label) where sequence shape is (T, E)."""
        seq   = torch.FloatTensor(self.sequences[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label


# ---------------------------------------------------------------------------
# Data loader factory
# ---------------------------------------------------------------------------

def create_data_loaders(
    processed_dir: Path,
    sequence_length: int = 128,
    stride: int = 16,
    batch_size: int = 32,
    num_workers: int = 0,
    max_files: Optional[int] = None,
    max_samples_per_file: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a single processed directory.

    Files are split by day-of-year (extracted from the ELS filename) so that
    no orbit crossing appears in more than one split — roadmap Step 7:
      Train : days   1 – 270  (Jan – Sep)
      Val   : days 271 – 335  (Oct – Nov)
      Test  : days 336 – 366  (Dec)

    Normalisation statistics are computed on the training set only, then
    applied to val and test to prevent data leakage.

    Returns:
        train_loader, val_loader, test_loader
    """
    all_files = sorted(processed_dir.rglob("*.nc"))
    if not all_files:
        raise ValueError(f"No .nc files found in {processed_dir}")

    if max_files:
        all_files = all_files[:max_files]

    # Split by day-of-year
    train_files, val_files, test_files = [], [], []
    skipped = 0
    for f in all_files:
        doy = day_of_year_from_path(f)
        if doy == -1:
            skipped += 1
            continue
        if doy <= 270:
            train_files.append(f)
        elif doy <= 335:
            val_files.append(f)
        else:
            test_files.append(f)

    print(f"Files found : {len(all_files)}  (skipped {skipped} with unrecognised names)")
    print(f"Train files : {len(train_files)}  (days   1–270)")
    print(f"Val   files : {len(val_files)}    (days 271–335)")
    print(f"Test  files : {len(test_files)}   (days 336–366)")

    # Build training dataset first — fit normalisation on training data only
    print("\nBuilding training dataset (fitting normalisation)...")
    train_ds = BowShockDataset(
        train_files,
        sequence_length=sequence_length,
        stride=stride,
        max_samples_per_file=max_samples_per_file,
    )

    # Val and test reuse the training mean/std — no leakage
    print("Building val / test datasets...")
    val_ds = BowShockDataset(
        val_files,
        sequence_length=sequence_length,
        stride=stride,
        normalize_mean=train_ds.mean,
        normalize_std=train_ds.std,
        max_samples_per_file=max_samples_per_file,
    )
    test_ds = BowShockDataset(
        test_files,
        sequence_length=sequence_length,
        stride=stride,
        normalize_mean=train_ds.mean,
        normalize_std=train_ds.std,
        max_samples_per_file=max_samples_per_file,
    )

    print(f"\nDataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    n_crossing = int((train_ds.labels == 1).sum())
    n_none     = int((train_ds.labels == 0).sum())
    print(f"Train label distribution — crossing: {n_crossing}, none: {n_none}\n")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )
