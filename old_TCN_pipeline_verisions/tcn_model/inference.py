"""
Inference script for TCN model to predict bow shock crossing events.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr

from tcn_model import BowShockTCN
from data_loader import load_nc_data, get_bow_shock_times, parse_yaml_labels


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    num_energy_bins: int = 63,
) -> BowShockTCN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model args
    args = checkpoint.get("args", {})
    num_channels = args.get("num_channels", [64, 128, 256, 128])
    dropout = args.get("dropout", 0.2)
    
    # Create model
    model = BowShockTCN(
        num_energy_bins=num_energy_bins,
        num_channels=num_channels,
        dropout=dropout,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def predict_sequences(
    model: BowShockTCN,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Predict bow shock events for sequences.
    
    Args:
        model: Trained TCN model
        sequences: Array of shape (n_sequences, sequence_length, num_energy_bins)
        device: Device to run inference on
        batch_size: Batch size for inference
        threshold: Classification threshold
    
    Returns:
        Predictions array of shape (n_sequences, sequence_length)
    """
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            # Get predictions
            outputs = model(batch_tensor)
            outputs = outputs.squeeze(-1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            all_preds.append(preds)
    
    return np.concatenate(all_preds, axis=0)


def predict_file(
    model: BowShockTCN,
    nc_path: Path,
    device: torch.device,
    sequence_length: int = 100,
    stride: int = 10,
    threshold: float = 0.5,
    normalize_mean: Optional[float] = None,
    normalize_std: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict bow shock events for a single NetCDF file.
    
    Args:
        model: Trained TCN model
        nc_path: Path to NetCDF file
        device: Device to run inference on
        sequence_length: Length of input sequences
        stride: Stride for sliding window
        threshold: Classification threshold
        normalize_mean: Mean for normalization (if None, compute from data)
        normalize_std: Std for normalization (if None, compute from data)
    
    Returns:
        Tuple of (times, predictions, probabilities)
        - times: Array of timestamps
        - predictions: Binary predictions (time,)
        - probabilities: Prediction probabilities (time,)
    """
    # Load data
    counts, times = load_nc_data(nc_path)
    
    # Normalize
    if normalize_mean is None:
        normalize_mean = np.nanmean(counts)
    if normalize_std is None:
        normalize_std = np.nanstd(counts)
        if normalize_std < 1e-8:
            normalize_std = 1.0
    
    counts = (counts - normalize_mean) / normalize_std
    counts = np.nan_to_num(counts, nan=0.0)
    
    # Create sequences
    sequences = []
    sequence_starts = []
    
    for start_idx in range(0, len(counts) - sequence_length + 1, stride):
        end_idx = start_idx + sequence_length
        sequence = counts[start_idx:end_idx]
        sequences.append(sequence)
        sequence_starts.append(start_idx)
    
    if len(sequences) == 0:
        # If file is shorter than sequence_length, pad it
        if len(counts) < sequence_length:
            padding = np.zeros((sequence_length - len(counts), counts.shape[1]))
            counts = np.vstack([counts, padding])
            sequences = [counts]
            sequence_starts = [0]
        else:
            # Use last sequence_length samples
            sequences = [counts[-sequence_length:]]
            sequence_starts = [len(counts) - sequence_length]
    
    sequences = np.array(sequences, dtype=np.float32)
    
    # Predict
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for sequence in sequences:
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            output = model(sequence_tensor)
            output = output.squeeze(-1).squeeze(0)
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            all_preds.append(preds)
            all_probs.append(probs)
    
    # Aggregate predictions across overlapping windows
    # Use majority voting or max probability
    n_times = len(times)
    aggregated_preds = np.zeros(n_times, dtype=int)
    aggregated_probs = np.zeros(n_times, dtype=float)
    counts_per_time = np.zeros(n_times, dtype=int)
    
    for seq_idx, start_idx in enumerate(sequence_starts):
        end_idx = min(start_idx + sequence_length, n_times)
        seq_len = end_idx - start_idx
        
        aggregated_preds[start_idx:end_idx] += all_preds[seq_idx][:seq_len]
        aggregated_probs[start_idx:end_idx] += all_probs[seq_idx][:seq_len]
        counts_per_time[start_idx:end_idx] += 1
    
    # Average probabilities
    mask = counts_per_time > 0
    aggregated_probs[mask] /= counts_per_time[mask]
    
    # Majority vote for predictions
    aggregated_preds = (aggregated_probs > threshold).astype(int)
    
    return times, aggregated_preds, aggregated_probs


def plot_predictions(
    nc_path: Path,
    times: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    yaml_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    """Plot predictions with ground truth labels if available."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Load spectrogram data for visualization
    ds = xr.open_dataset(nc_path)
    counts = ds["count_rate"].values.astype(float)
    counts[counts == 65535.0] = np.nan
    counts = np.nanmean(counts, axis=2)  # Average over anodes
    counts = np.nan_to_num(counts, nan=0.0)
    
    times_dt = pd.to_datetime(times).to_pydatetime()
    
    # Plot spectrogram
    ax1 = axes[0]
    energy = ds["energy"].values
    if energy.ndim == 1:
        # Create 2D energy grid
        energy_2d = np.tile(energy, (len(times), 1))
    else:
        energy_2d = energy
    
    # Plot spectrogram
    im = ax1.pcolormesh(
        times_dt,
        energy_2d[0] if energy_2d.ndim == 2 else energy,
        counts.T,
        shading="auto",
        cmap="viridis",
        norm=plt.colors.LogNorm(vmin=max(1.0, counts[counts > 0].min()), vmax=counts.max()),
    )
    ax1.set_ylabel("Energy (eV/q)")
    ax1.set_yscale("log")
    ax1.set_title(f"Spectrogram: {nc_path.name}")
    plt.colorbar(im, ax=ax1, label="Counts / s")
    
    # Add ground truth bow shock events
    if yaml_path and yaml_path.exists():
        bow_shock_times = get_bow_shock_times(yaml_path)
        for bs_time in bow_shock_times:
            ax1.axvline(bs_time, color="red", linewidth=2, linestyle="--", alpha=0.7, label="True Bow Shock")
    
    # Plot predictions
    ax2 = axes[1]
    ax2.plot(times_dt, probabilities, label="Probability", color="blue", alpha=0.7)
    ax2.fill_between(times_dt, 0, predictions, alpha=0.3, color="red", label="Predicted Events")
    ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax2.set_ylabel("Probability / Prediction")
    ax2.set_xlabel("Time")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y/%H:%M"))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    ds.close()


def main():
    parser = argparse.ArgumentParser(description="Predict bow shock events using trained TCN model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input NetCDF file or directory",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory containing YAML label files (for visualization). If not specified, uses relative path from project root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (if None, display plots)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Length of input sequences",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device, num_energy_bins=63)
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        nc_files = [input_path]
    else:
        nc_files = sorted(input_path.rglob("*.nc"))
    
    if len(nc_files) == 0:
        raise ValueError(f"No NetCDF files found in {input_path}")
    
    print(f"Processing {len(nc_files)} file(s)...")
    
    # Create output directory if needed
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default labels directory if not provided
    if args.labels_dir:
    labels_dir = Path(args.labels_dir)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        labels_dir = project_root / "data" / "zenodo-3946033" / "crossings" / "labels" / "all"
    
    # Process each file
    for nc_path in nc_files:
        print(f"\nProcessing: {nc_path.name}")
        
        try:
            # Predict
            times, predictions, probabilities = predict_file(
                model,
                nc_path,
                device,
                sequence_length=args.sequence_length,
                stride=args.stride,
                threshold=args.threshold,
            )
            
            # Print summary
            n_events = np.sum(predictions == 1)
            print(f"  Predicted {n_events} bow shock events")
            
            # Get ground truth if available
            yaml_path = labels_dir / f"{nc_path.stem}.yaml"
            if yaml_path.exists():
                bow_shock_times = get_bow_shock_times(yaml_path)
                print(f"  Ground truth: {len(bow_shock_times)} bow shock events")
            
            # Plot
            output_path = None
            if output_dir:
                output_path = output_dir / f"{nc_path.stem}_predictions.png"
            
            plot_predictions(
                nc_path,
                times,
                predictions,
                probabilities,
                yaml_path if yaml_path.exists() else None,
                output_path,
            )
        
        except Exception as e:
            print(f"  Error processing {nc_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()

