"""
Data loader for TCN model training on bow shock crossing events.

Labels come from the crossing_event variable inside each .nc file:
  0 → no crossing event
  1 → crossing event (bow shock or magnetopause)

All .nc files live in a single processed directory (new_processed/).
The train/val/test split is done by day-of-year extracted from the filename,
matching the roadmap split: train days 1-270, val 271-335, test 336-366.

Class balance strategy
----------------------
Each .nc file that contains a crossing event gets two kinds of windows:

  1. Crossing-centred sliding windows
     A context region is built around each crossing event midpoint:
       [ midpoint - context_hours,  midpoint + context_hours ]
     A sliding window of length `sequence_length` moves across this region
     with step `crossing_stride`.  Every window whose centre timestep falls
     inside the crossing label region gets label 1; all others get label 0.
     Using a smaller crossing_stride here gives dense, overlapping samples
     so the model sees the event from many angles.

  2. Background sliding windows
     Standard sliding window with `background_stride` over the whole file.
     Label = centre-timestep label (almost always 0).

Both window sizes are fixed at `sequence_length` so every sample that
enters the model is the same shape — only the stride and region differ.

Key parameters you can tune at runtime
---------------------------------------
  --context-hours       Hours of data each side of crossing midpoint (default 24)
  --crossing-stride     Slide step inside the context region (default 4)
  --sequence-length     Fixed window length fed to the model (default 128)
  --stride              Slide step for background windows (default 16)
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
# Cadence constant — ELS records roughly one sample every 8 seconds
# ---------------------------------------------------------------------------
SECONDS_PER_TIMESTEP = 8


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
        counts : np.ndarray (time, energy) — log-scaled energy data
        labels : np.ndarray (time,)        — 0/1 per timestep
        times  : np.ndarray of datetime64 timestamps
    """
    ds = xr.open_dataset(nc_path)

    # --- Count rate -------------------------------------------------------
    counts = ds["count_rate"].values.astype(float)
    counts[counts == 65535.0] = np.nan

    if counts.ndim == 3:
        counts = np.nanmean(counts, axis=2)     # average over anodes → (time, energy)

    counts = np.clip(counts, 0.0, None)         # clip instrument artefacts

    # Interpolate NaN along time for each energy channel
    for ch in range(counts.shape[1]):
        col    = counts[:, ch]
        finite = np.isfinite(col)
        if finite.sum() >= 2 and (~finite).any():
            counts[~finite, ch] = np.interp(
                np.where(~finite)[0],
                np.where(finite)[0],
                col[finite],
            )

    counts = np.nan_to_num(counts, nan=0.0)
    counts = np.log10(counts + 1e-6)            # log-scale (roadmap Step 4a)

    # --- Labels -----------------------------------------------------------
    if "crossing_event" not in ds:
        raise ValueError(
            f"{nc_path.name} has no 'crossing_event' variable. "
            "Run add_crossing_event_new.py first."
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
    Returns -1 if pattern is not found (file will be excluded from splits).
    """
    match = re.search(r"ELS_\d{4}(\d{3})\d{2}_", nc_path.name)
    if match:
        return int(match.group(1))
    return -1


# ---------------------------------------------------------------------------
# Helper — convert hours to timesteps
# ---------------------------------------------------------------------------

def hours_to_steps(hours: float) -> int:
    """Convert hours to number of timesteps at the ELS ~8 s cadence."""
    return int(hours * 3600 / SECONDS_PER_TIMESTEP)


# ---------------------------------------------------------------------------
# Crossing-event detector
# ---------------------------------------------------------------------------

def find_crossing_events(labels: np.ndarray) -> list[dict]:
    """
    Scan a label array and return one dict per distinct crossing event.

    Consecutive label-1 timesteps are grouped into a single event.
    Non-consecutive label-1 timesteps (gap > 1) are treated as separate events.

    Returns a list of dicts:
        { "start": int, "end": int, "midpoint": int }
    where start/end are inclusive indices and midpoint is the centre index.
    """
    crossing_idx = np.where(labels == 1)[0]
    if len(crossing_idx) == 0:
        return []

    events = []
    group  = [crossing_idx[0]]

    for idx in crossing_idx[1:]:
        if idx == group[-1] + 1:        # consecutive — same event
            group.append(idx)
        else:                           # gap — start a new event
            events.append(group)
            group = [idx]
    events.append(group)

    return [
        {
            "start":    g[0],
            "end":      g[-1],
            "midpoint": g[len(g) // 2],
        }
        for g in events
    ]


# ---------------------------------------------------------------------------
# Crossing-centred sliding window builder
# ---------------------------------------------------------------------------

def build_crossing_windows(
    counts: np.ndarray,
    labels: np.ndarray,
    sequence_length: int,
    context_hours: float = 24.0,
    crossing_stride: int = 4,
) -> Tuple[list[np.ndarray], list[int]]:
    """
    Slide a fixed-length window across the context region around each crossing.

    For each crossing event found in `labels`:
      1. Find the event midpoint (centre of the label-1 block)
      2. Define a context region:
             [ midpoint - context_steps,  midpoint + context_steps ]
         clipped to the file boundaries
      3. Slide a window of length `sequence_length` across this region
         using `crossing_stride` as the step size
      4. Label each window by its own centre timestep (1 if crossing, 0 if not)

    Using a small crossing_stride (e.g. 4 ≈ 32 s) creates many overlapping
    windows around each event so the model sees the full approach and departure.

    Args:
        counts          : (time, energy) log-scaled array for this file
        labels          : (time,) integer label array for this file
        sequence_length : fixed window length — same as model input size
        context_hours   : hours each side of the crossing midpoint to cover
        crossing_stride : timestep advance between windows in the context region

    Returns:
        sequences  : list of np.ndarray, each shape (sequence_length, energy)
        win_labels : list of int — 1 if window centre is a crossing, else 0
    """
    context_steps = hours_to_steps(context_hours)
    n             = len(counts)
    half          = sequence_length // 2

    events = find_crossing_events(labels)
    if not events:
        return [], []

    sequences  = []
    win_labels = []

    for event in events:
        midpoint = event["midpoint"]

        # Context region — clipped to file boundaries
        region_start = max(0, midpoint - context_steps)
        region_end   = min(n, midpoint + context_steps)

        # Slide window across the context region
        for win_start in range(region_start, region_end - sequence_length + 1, crossing_stride):
            win_end    = win_start + sequence_length
            win_centre = win_start + half

            if win_end > n:
                break

            win_label = int(labels[win_centre])
            sequences.append(counts[win_start:win_end].copy())
            win_labels.append(win_label)

    return sequences, win_labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BowShockDataset(Dataset):
    """
    Sliding-window dataset over NetCDF files with crossing-centred sampling.

    For every file two passes are made:

      Pass 1 — Crossing-centred sliding windows
        A dense sliding window (step = crossing_stride) sweeps across a
        ±context_hours region around each crossing midpoint.
        Window label = centre-timestep label (1 inside the crossing, 0 outside).

      Pass 2 — Background sliding windows
        Standard sliding window (step = background_stride) over the whole file.
        Label = centre-timestep label (almost always 0).

    All windows are exactly `sequence_length` timesteps long so every
    sample entering the model is the same shape.

    Args:
        nc_files             : list of NetCDF file paths to load
        sequence_length      : fixed number of timesteps per window
        background_stride    : slide step for the whole-file background pass
        context_hours        : hours each side of crossing midpoint to cover
        crossing_stride      : slide step inside the crossing context region
        normalize_mean       : per-energy-bin mean  (fit on training data only)
        normalize_std        : per-energy-bin std   (fit on training data only)
        max_samples_per_file : cap background windows per file
    """

    def __init__(
        self,
        nc_files: list[Path],
        sequence_length: int    = 128,
        background_stride: int  = 16,
        context_hours: float    = 24.0,
        crossing_stride: int    = 4,
        normalize_mean: Optional[np.ndarray] = None,
        normalize_std:  Optional[np.ndarray] = None,
        max_samples_per_file: Optional[int]  = None,
    ):
        self.sequences: list[np.ndarray] = []
        self.labels:    list[int]        = []

        bg_centre = sequence_length // 2

        for nc_path in nc_files:
            try:
                counts, file_labels, _ = load_nc_data(nc_path)
            except Exception as e:
                print(f"  Warning: skipping {nc_path.name} — {e}")
                continue

            n = len(counts)

            # ----------------------------------------------------------
            # Pass 1 — dense crossing-centred windows
            # ----------------------------------------------------------
            cross_seqs, cross_labs = build_crossing_windows(
                counts,
                file_labels,
                sequence_length,
                context_hours,
                crossing_stride,
            )
            self.sequences.extend(cross_seqs)
            self.labels.extend(cross_labs)

            # ----------------------------------------------------------
            # Pass 2 — sparse background windows over the whole file
            # ----------------------------------------------------------
            added = 0
            for start in range(0, n - sequence_length + 1, background_stride):
                win_label = int(file_labels[start + bg_centre])
                self.sequences.append(counts[start : start + sequence_length].copy())
                self.labels.append(win_label)
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
        # Fit only on training data — pass mean/std for val and test
        if normalize_mean is None or normalize_std is None:
            flat           = self.sequences.reshape(-1, self.sequences.shape[2])
            normalize_mean = flat.mean(axis=0)
            normalize_std  = flat.std(axis=0)
            normalize_std  = np.where(normalize_std < 1e-8, 1.0, normalize_std)

        self.mean      = normalize_mean
        self.std       = normalize_std
        self.sequences = (self.sequences - self.mean) / self.std
        self.sequences = np.nan_to_num(self.sequences, nan=0.0)

        # Report balance so you can decide whether to adjust strides
        n_cross = int((self.labels == 1).sum())
        n_none  = int((self.labels == 0).sum())
        ratio   = n_none / max(n_cross, 1)
        print(f"  Dataset built — background: {n_none}, crossing: {n_cross}  "
              f"(ratio {ratio:.1f}:1)")

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
    sequence_length: int   = 128,
    background_stride: int = 16,
    context_hours: float   = 24.0,
    crossing_stride: int   = 4,
    batch_size: int        = 32,
    num_workers: int       = 0,
    max_files: Optional[int]            = None,
    max_samples_per_file: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a single processed directory.

    Files are split by day-of-year so no orbit crossing leaks across splits:
      Train : days   1 – 270  (Jan – Sep)
      Val   : days 271 – 335  (Oct – Nov)
      Test  : days 336 – 366  (Dec)

    Normalisation statistics are fit on the training set only, then applied
    to val and test to prevent data leakage.

    Key parameters
    --------------
    context_hours     : how far each side of a crossing midpoint to sample
                        e.g. 24.0 means 24 h before AND 24 h after
    crossing_stride   : window slide step inside the context region
                        smaller = more windows per crossing = better balance
                        e.g. 4 timesteps ≈ 32 s between windows
    background_stride : window slide step for the whole-file background pass
                        larger = fewer background windows = better balance
                        e.g. 16 timesteps ≈ 2 min between windows

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

    print(f"Files found       : {len(all_files)}  (skipped {skipped} with unrecognised names)")
    print(f"Train files       : {len(train_files)}  (days   1–270)")
    print(f"Val   files       : {len(val_files)}    (days 271–335)")
    print(f"Test  files       : {len(test_files)}   (days 336–366)")
    print(f"Context           : ±{context_hours}h around each crossing midpoint")
    print(f"Crossing stride   : {crossing_stride} timesteps (~{crossing_stride * SECONDS_PER_TIMESTEP}s)")
    print(f"Background stride : {background_stride} timesteps (~{background_stride * SECONDS_PER_TIMESTEP}s)")

    # Build training dataset first — fits normalisation
    print("\nBuilding training dataset...")
    train_ds = BowShockDataset(
        train_files,
        sequence_length      = sequence_length,
        background_stride    = background_stride,
        context_hours        = context_hours,
        crossing_stride      = crossing_stride,
        max_samples_per_file = max_samples_per_file,
    )

    # Val and test reuse training mean/std — no leakage
    print("Building val / test datasets...")
    val_ds = BowShockDataset(
        val_files,
        sequence_length      = sequence_length,
        background_stride    = background_stride,
        context_hours        = context_hours,
        crossing_stride      = crossing_stride,
        normalize_mean       = train_ds.mean,
        normalize_std        = train_ds.std,
        max_samples_per_file = max_samples_per_file,
    )
    test_ds = BowShockDataset(
        test_files,
        sequence_length      = sequence_length,
        background_stride    = background_stride,
        context_hours        = context_hours,
        crossing_stride      = crossing_stride,
        normalize_mean       = train_ds.mean,
        normalize_std        = train_ds.std,
        max_samples_per_file = max_samples_per_file,
    )

    print(f"\nDataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    n_crossing = int((train_ds.labels == 1).sum())
    n_none     = int((train_ds.labels == 0).sum())
    pos_weight = n_none / max(n_crossing, 1)
    print(f"Train labels      — crossing: {n_crossing}, none: {n_none}")
    print(f"Recommended pos_weight for BCEWithLogitsLoss = {pos_weight:.1f}\n")

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = torch.cuda.is_available(),
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )
