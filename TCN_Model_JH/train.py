"""
Training script for the TCN bow shock model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from data_loader import create_data_loaders
from tcn_model import BowShockTCN


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Flatten arrays and return accuracy / precision / recall / F1."""
    t, p = y_true.flatten(), y_pred.flatten()
    return {
        "accuracy":  float(accuracy_score(t, p)),
        "precision": float(precision_score(t, p, zero_division=0)),
        "recall":    float(recall_score(t, p, zero_division=0)),
        "f1":        float(f1_score(t, p, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> Dict[str, float]:
    """
    One pass over a DataLoader.

    If optimizer is provided the model is trained; otherwise it is evaluated.
    Labels are per-sequence scalars (0 or 1) derived from the centre timestep.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for sequences, labels in loader:
            sequences = sequences.to(device)       # (B, T, E)
            labels    = labels.float().to(device)  # (B,)

            # Model output: (B, T, 1) → average over time → (B,)
            logits    = model(sequences).squeeze(-1)   # (B, T)
            preds_seq = logits.mean(dim=1)             # (B,)

            loss = criterion(preds_seq, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds_binary = (torch.sigmoid(preds_seq) > 0.5).long().cpu().numpy()
            all_preds.append(preds_binary)
            all_labels.append(labels.long().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description="Train TCN for bow shock detection")

    # ---- Data ----
    parser.add_argument(
        "--data-dir",
        default=str(project_root / "data" / "new_processed"),
        help="Path to new_processed/ directory containing labelled .nc files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "checkpoints"),
    )

    # ---- Window ----
    parser.add_argument("--sequence-length", type=int,   default=128)
    parser.add_argument("--stride",          type=int,   default=16)

    # ---- Training ----
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--lr",              type=float, default=1e-3)

    # ---- Model ----
    parser.add_argument("--dropout",         type=float, default=0.2)
    parser.add_argument("--num-channels",    type=int, nargs="+", default=[64, 128, 256, 128])

    # ---- Misc ----
    parser.add_argument("--max-files",             type=int, default=None)
    parser.add_argument("--max-samples-per-file",  type=int, default=None)
    parser.add_argument("--device",                default="auto")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data -------------------------------------------------------------
    train_loader, val_loader, test_loader = create_data_loaders(
        processed_dir        = Path(args.data_dir),
        sequence_length      = args.sequence_length,
        stride               = args.stride,
        batch_size           = args.batch_size,
        max_files            = args.max_files,
        max_samples_per_file = args.max_samples_per_file,
    )

    # --- Model ------------------------------------------------------------
    sample_seq, _ = next(iter(train_loader))
    num_energy_bins = sample_seq.shape[2]

    model = BowShockTCN(
        num_energy_bins = num_energy_bins,
        num_channels    = args.num_channels,
        dropout         = args.dropout,
    ).to(device)

    print(f"Energy bins : {num_energy_bins}")
    print(f"Parameters  : {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- Training loop ----------------------------------------------------
    best_f1 = 0.0
    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_m   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step(val_m["loss"])

        history["train"].append(train_m)
        history["val"].append(val_m)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss {train_m['loss']:.4f}  f1 {train_m['f1']:.4f} | "
            f"val   loss {val_m['loss']:.4f}  f1 {val_m['f1']:.4f}  "
            f"prec {val_m['precision']:.4f}  rec {val_m['recall']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_m,
                    "args": vars(args),
                },
                output_dir / "best_model.pt",
            )
            print(f"  → saved best model (val F1 {best_f1:.4f})")

    # --- Test evaluation --------------------------------------------------
    test_m = run_epoch(model, test_loader, criterion, device)
    print(f"\nTest: {test_m}")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "test_metrics": test_m,
            "args": vars(args),
        },
        output_dir / "final_model.pt",
    )

    history["test"] = test_m
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
