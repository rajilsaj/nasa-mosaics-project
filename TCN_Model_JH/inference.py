"""
Inference script — predict bow shock crossings on preprocessed .npy files.

Usage:
    python inference.py --model checkpoints/best_model.pt --input path/to/file_X_63.npy
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
import joblib

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
        num_channels    = saved_args.get("num_channels", [32, 64, 128, 64, 32]),
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
    X: np.ndarray,         # shape (time, 63)
    t_ns: np.ndarray,      # microsecond timestamps
    device: torch.device,
    sequence_length: int = 100,
    stride: int = 10,
    threshold: float = 0.5,
    normalize_mean: Optional[np.ndarray] = None,
    normalize_std:  Optional[np.ndarray] = None,
    scaler_path=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    times = pd.to_datetime(t_ns, unit="us", utc=True)   # define times from t_ns
    # Per-energy-bin normalization
    if normalize_mean is None:
        scaler = joblib.load(Path(scaler_path))  # load from CLI arg
        normalize_mean = scaler.mean_
        normalize_std = scaler.scale_

    counts = (X - normalize_mean) / normalize_std
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
    file_key: str,
    X: np.ndarray,
    times: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    output_path=None,
):
    # X shape: (time, 63) — already loaded, no file access needed
    times_dt = times.to_pydatetime()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Spectrogram from npy array
    vmin = max(1e-6, X[X > 0].min()) if (X > 0).any() else 1e-6
    im = ax1.pcolormesh(
        times_dt, np.arange(63), X.T,
        shading="auto", cmap="viridis",
    )
    ax1.set_ylabel("Energy bin")
    ax1.set_title(f"Spectrogram — {file_key}")
    plt.colorbar(im, ax=ax1, label="log10 counts")

    # Probabilities / predictions (unchanged)
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
    parser.add_argument("--scaler", required=True, help="Path to scaler.pkl from fit_scaler.py")
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
    npy_files = sorted(input_path.rglob("*_X_63.npy")) if input_path.is_dir() else [input_path]
    if not npy_files:
        raise ValueError(f"No _X_63.npy files found in {input_path}")

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for x_path in npy_files:
        file_key = x_path.stem.replace("_X_63", "")
        t_path = x_path.parent / f"{file_key}_t_ns.npy"

        if not t_path.exists():
            print(f"  Skipping {file_key} — missing t_ns file")
            continue

        print(f"\nProcessing {file_key}")
        try:
            X = np.load(x_path)
            t_ns = np.load(t_path)

            times, preds, probs = predict_file(
                model, X, t_ns, device,
                sequence_length=args.sequence_length,
                stride=args.stride,
                threshold=args.threshold,
            )

            print(f"  Predicted events at {preds.sum()} time steps")
            out = output_dir / f"{file_key}_predictions.png" if output_dir else None
            plot_predictions(file_key, X, times, preds, probs, out)

        except Exception as e:
            print(f"  Error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
