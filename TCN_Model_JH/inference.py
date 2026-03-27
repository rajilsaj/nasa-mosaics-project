"""
Inference script — predict bow shock crossings on new NetCDF files.

Usage:
    python inference.py --model checkpoints/best_model.pt --input path/to/file.nc
    python inference.py --model checkpoints/best_model.pt --input path/to/folder/
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from data_loader import load_nc_data
from tcn_model import BowShockTCN


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: torch.device) -> tuple[BowShockTCN, dict]:
    """Load a trained model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    model = BowShockTCN(
        num_energy_bins = saved_args.get("num_energy_bins", 63),
        num_channels    = saved_args.get("num_channels", [64, 128, 256, 128]),
        dropout         = saved_args.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, saved_args


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_file(
    model: BowShockTCN,
    nc_path: Path,
    device: torch.device,
    sequence_length: int = 100,
    stride: int = 10,
    threshold: float = 0.5,
    normalize_mean: Optional[np.ndarray] = None,
    normalize_std:  Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model on a single NetCDF file using a sliding window.

    Normalization uses per-energy-bin arrays (same as training).
    If not provided they are estimated from this file alone — less accurate,
    so pass the training stats when available.

    Returns:
        times       : datetime64 array  (n_time,)
        predictions : binary int array  (n_time,)
        probs       : float array       (n_time,)
    """
    counts, _, times = load_nc_data(nc_path)  # _ discards labels, we don't need them at inference

    # Per-energy-bin normalization
    if normalize_mean is None:
        normalize_mean = counts.mean(axis=0)
        normalize_std  = counts.std(axis=0)
        normalize_std  = np.where(normalize_std < 1e-8, 1.0, normalize_std)

    counts = (counts - normalize_mean) / normalize_std
    counts = np.nan_to_num(counts, nan=0.0)

    # Build sliding windows
    starts = list(range(0, len(counts) - sequence_length + 1, stride))
    if not starts:
        # File shorter than one window — pad with zeros
        pad    = np.zeros((sequence_length - len(counts), counts.shape[1]))
        counts = np.vstack([counts, pad])
        starts = [0]

    sequences = np.stack([counts[s : s + sequence_length] for s in starts]).astype(np.float32)

    # Run model
    prob_sum   = np.zeros(len(times))
    count_hits = np.zeros(len(times), dtype=int)

    with torch.no_grad():
        for i, start in enumerate(starts):
            seq_tensor = torch.FloatTensor(sequences[i]).unsqueeze(0).to(device)
            logits = model(seq_tensor).squeeze(-1).squeeze(0)   # (T,)
            probs  = torch.sigmoid(logits).cpu().numpy()

            end = min(start + sequence_length, len(times))
            prob_sum[start:end]   += probs[: end - start]
            count_hits[start:end] += 1

    # Average overlapping windows
    mask = count_hits > 0
    prob_sum[mask] /= count_hits[mask]

    predictions = (prob_sum > threshold).astype(int)
    return times, predictions, prob_sum


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_predictions(
    nc_path: Path,
    times: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path: Optional[Path] = None,
):
    """Plot the spectrogram alongside the model's prediction probabilities."""
    ds     = xr.open_dataset(nc_path)
    counts = ds["count_rate"].values.astype(float)
    counts[counts == 65535.0] = np.nan
    if counts.ndim == 3:
        counts = np.nanmean(counts, axis=2)
    counts = np.nan_to_num(counts, nan=0.0)

    energy    = ds["energy"].values
    times_dt  = pd.to_datetime(times).to_pydatetime()
    ds.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Spectrogram
    vmin = max(1.0, counts[counts > 0].min()) if (counts > 0).any() else 1.0
    im = ax1.pcolormesh(
        times_dt, energy, counts.T,
        shading="auto", cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=counts.max()),
    )
    ax1.set_ylabel("Energy (eV/q)")
    ax1.set_yscale("log")
    ax1.set_title(f"Spectrogram — {nc_path.name}")
    plt.colorbar(im, ax=ax1, label="Counts / s")

    # Probabilities / predictions
    ax2.plot(times_dt, probabilities, color="steelblue", alpha=0.8, label="Probability")
    ax2.fill_between(times_dt, 0, predictions, alpha=0.25, color="red", label="Predicted event")
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold 0.5")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Time")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y %H:%M"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {output_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bow shock inference with trained TCN")
    parser.add_argument("--model",           required=True,  help="Path to checkpoint (.pt)")
    parser.add_argument("--input",           required=True,  help="NetCDF file or directory")
    parser.add_argument("--output",          default=None,   help="Directory to save plots")
    parser.add_argument("--sequence-length", type=int,   default=100)
    parser.add_argument("--stride",          type=int,   default=10)
    parser.add_argument("--threshold",       type=float, default=0.5)
    parser.add_argument("--device",          default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    checkpoint_path = Path(args.model)
    model, _ = load_model(checkpoint_path, device)
    print(f"Loaded model from {checkpoint_path}")

    input_path = Path(args.input)
    nc_files   = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.nc"))
    if not nc_files:
        raise ValueError(f"No .nc files found in {input_path}")

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for nc_path in nc_files:
        print(f"\nProcessing {nc_path.name}")
        try:
            times, preds, probs = predict_file(
                model, nc_path, device,
                sequence_length=args.sequence_length,
                stride=args.stride,
                threshold=args.threshold,
            )
            print(f"  Predicted events at {preds.sum()} time steps")
            out = output_dir / f"{nc_path.stem}_predictions.png" if output_dir else None
            plot_predictions(nc_path, times, preds, probs, out)
        except Exception as e:
            print(f"  Error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
