"""
Data loader for TCN model training — reads pre-built .npy window files.

Expected files in data_dir:
    train_X.npy  shape (N, 128, 63)   float32
    train_y.npy  shape (N,)            uint8
    val_X.npy    shape (N, 128, 63)
    val_y.npy    shape (N,)
    test_X.npy   shape (N, 128, 63)
    test_y.npy   shape (N,)

These are produced by windowbuilder.py.  The windowing and train/val/test
split are already done — this loader just normalises and wraps in DataLoaders.

Normalisation
-------------
Done by the preprocess pipeline
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """
    Thin wrapper around pre-built numpy window arrays.

    Args:
        X    : float32 array of shape (N, time_steps, energy_bins)
        y    : int array of shape (N,)
        mean : per-energy-bin mean  — fit on training data only
        std  : per-energy-bin std   — fit on training data only
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        self.y = y.astype(np.int64)

        # Report class balance so you can spot imbalance problems early
        n_cross = int((self.y == 1).sum())
        n_none  = int((self.y == 0).sum())
        ratio   = n_none / max(n_cross, 1)
        print(f"  Dataset built — background: {n_none}, crossing: {n_cross} "
              f"(ratio {ratio:.1f}:1)")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_data_loaders(
    data_dir: Path,
    batch_size:  int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load pre-built window arrays and return train / val / test DataLoaders.

    Args:
        data_dir    : folder containing train_X.npy, train_y.npy, etc.
        batch_size  : sequences per training batch
        num_workers : parallel workers for data loading (0 = main process)

    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)

    # ---- Load arrays -------------------------------------------------
    print(f"Loading data from {data_dir}")
    train_X = np.load(data_dir / "train_X.npy")
    train_y = np.load(data_dir / "train_y.npy")
    val_X   = np.load(data_dir / "val_X.npy")
    val_y   = np.load(data_dir / "val_y.npy")
    test_X  = np.load(data_dir / "test_X.npy")
    test_y  = np.load(data_dir / "test_y.npy")

    print(f"Train windows : {len(train_X)}")
    print(f"Val   windows : {len(val_X)}")
    print(f"Test  windows : {len(test_X)}")

    # ---- Build datasets — val/test reuse training stats --------------
    print("\nBuilding training dataset...")
    train_ds = WindowDataset(train_X, train_y)

    print("Building val dataset...")
    val_ds = WindowDataset(val_X, val_y)

    print("Building test dataset...")
    test_ds = WindowDataset(test_X, test_y)

    # ---- Class balance info for loss weighting -----------------------
    n_none     = int((train_ds.y == 0).sum())
    n_crossing = int((train_ds.y == 1).sum())
    raw_weight = n_none / max(n_crossing, 1)
    print(f"\nTrain labels      — crossing: {n_crossing}, none: {n_none}")
    print(f"Recommended pos_weight for BCEWithLogitsLoss = {raw_weight:.1f}\n")

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
