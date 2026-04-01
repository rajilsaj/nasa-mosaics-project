"""
Training script for TCN model to detect bow shock crossing events.
Optimized for small_tcn_model with faster defaults for testing.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tcn_model import BowShockTCN
from data_loader import create_data_loaders


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Handle case where all predictions are the same
    if len(np.unique(y_pred_flat)) == 1:
        if y_pred_flat[0] == 0:
            precision = 0.0
            recall = 0.0
        else:
            precision = (y_true_flat == y_pred_flat).sum() / len(y_pred_flat)
            recall = 1.0 if (y_true_flat == 1).any() else 0.0
    else:
        precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)  # (batch, seq_len, 1)
        outputs = outputs.squeeze(-1)  # (batch, seq_len)
        
        # Compute loss
        loss = criterion(outputs, labels.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(train_loader)
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(sequences)
            outputs = outputs.squeeze(-1)
            
            # Compute loss
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            
            # Predictions
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    # Compute metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(val_loader)
    
    return metrics


def main():
    # Get script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    parser = argparse.ArgumentParser(
        description="Train TCN model for bow shock detection (tcn_model_2.0 - full model without constraints)"
    )
    parser.add_argument(
        "--true-data-dir",
        type=str,
        default=str(project_root / "data" / "true_processed"),
        help="Directory containing true processed NetCDF files (all labeled as positive/events)",
    )
    parser.add_argument(
        "--false-data-dir",
        type=str,
        default=str(project_root / "data" / "false_processed"),
        help="Directory containing false processed NetCDF files (no bow shock events, all negative)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(script_dir / "checkpoints"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        help="Length of input sequences (default: 100)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10,
        help="Stride for sliding window (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        nargs="+",
        default=[64, 128, 256, 128],
        help="Number of channels in each TCN layer (default: [64, 128, 256, 128])",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to use from each directory (default: None, uses all files)",
    )
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=None,
        help="Maximum number of sequences to create per file (default: None, uses all sequences)",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("=" * 60)
    print("TCN_MODEL_2.0 - Full Model Configuration")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    print("=" * 60)
    print("Training Configuration:")
    print(f"  True data (all positive): {args.true_data_dir}")
    print(f"  False data (all negative): {args.false_data_dir}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max files: {args.max_files if args.max_files else 'None (all files)'}")
    print(f"  Max samples per file: {args.max_samples_per_file if args.max_samples_per_file else 'None (all sequences)'}")
    print("=" * 60)
    print()
    
    train_loader, val_loader, test_loader = create_data_loaders(
        true_processed_dir=Path(args.true_data_dir),
        false_processed_dir=Path(args.false_data_dir),
        sequence_length=args.sequence_length,
        stride=args.stride,
        train_split=args.train_split,
        val_split=args.val_split,
        batch_size=args.batch_size,
        normalize=True,
        max_files=args.max_files,
        max_samples_per_file=args.max_samples_per_file,
    )
    
    # Get number of energy bins from first batch
    sample_seq, _ = next(iter(train_loader))
    num_energy_bins = sample_seq.shape[2]
    print(f"Number of energy bins: {num_energy_bins}")
    
    # Create model
    model = BowShockTCN(
        num_energy_bins=num_energy_bins,
        num_channels=args.num_channels,
        dropout=args.dropout,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    
    # Training loop
    best_val_f1 = 0.0
    train_history = []
    val_history = []
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        train_history.append(train_metrics)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        val_history.append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_metrics["loss"])
        
        # Print metrics
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )
        
        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  Saved best model (F1: {best_val_f1:.4f})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = validate(model, test_loader, criterion, device)
    print(f"Test Metrics: {test_metrics}")
    
    # Save final model and history
    final_checkpoint = {
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_metrics": test_metrics,
        "args": vars(args),
    }
    torch.save(final_checkpoint, output_dir / "final_model.pt")
    
    # Save training history
    history = {
        "train": train_history,
        "val": val_history,
        "test": test_metrics,
    }
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Models saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
