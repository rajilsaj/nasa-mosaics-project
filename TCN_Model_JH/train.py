"""
Training script for the TCN bow shock model.
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import joblib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from data_loader import create_data_loaders
from tcn_model import BowShockTCN

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix



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

    If optimizer is provided the model is trained (weighted loss).
    Otherwise the model is evaluated (unweighted loss — comparable numbers).
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

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
            preds_binary = (torch.sigmoid(preds_seq) > 0.3).long().cpu().numpy()   #was 0.5
            probs_np = torch.sigmoid(preds_seq).detach().cpu().numpy()
            all_probs.append(probs_np)
            all_preds.append(preds_binary)
            all_labels.append(labels.long().cpu().numpy())

    metrics = compute_metrics(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
    )
    metrics["loss"] = total_loss / len(loader)
    return metrics, np.concatenate(all_labels), np.concatenate(all_probs)

def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path):
    """
    Plots predicted probability vs actual frequency.
    A perfectly calibrated model follows the diagonal line.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0,1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=150)
    plt.close()
    print("saved calibration curve to:", output_dir / "calibration_curve.png")
    
def plot_roc_and_pr(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path):
    """
    ROC curve: shows tradeoff between true positive rate and false positive rate.
    PR curve:  shows tradeoff between precision and recall — more informative
               than ROC on imbalanced datasets like yours.
    """
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    # Precision-Recall
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC plot
    ax1.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC Curve")
    ax1.legend()

    # PR plot
    ax2.plot(rec, prec, label=f"AUC = {pr_auc:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()


    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curves.png", dpi=150)
    plt.close()

    print(f"ROC AUC:  {roc_auc:.4f}")
    print(f"PR AUC:   {pr_auc:.4f}")
    print("Saved ROC and PR curves to:", output_dir / "roc_pr_curves.png")
    
    return roc_auc, pr_auc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TCN for bow shock detection")

    # ---- Data ----
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to folder containing train_X.npy, train_y.npy, val_X.npy, etc.",
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Where to save model checkpoints",
    )

    # ---- Training ----
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)

    # ---- Model ----
    parser.add_argument("--dropout",      type=float, default=0.2)
    parser.add_argument("--num-channels", type=int, nargs="+", default=[64, 128, 256, 128])

    # ---- Misc ----
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dir = Path("home/jhuss/nasa-mosaics-project/data/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Data -------------------------------------------------------------
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir    = Path(args.data_dir),
        batch_size  = args.batch_size,
    )

    # ------ Weight from preporcess pipeline loader-------------------------
    # Loads the .pk1 file that is created from the pre preprocess pipeline
    class_weights = joblib.load(Path(args.data_dir) / "class_weights.pkl")
    pos_weight_val = min(class_weights[1], 20.0)  # Class 1 = crossing, cap at 50 was 10

    print(f"Loaded class weights: {class_weights}")
    print(f"Using pos_weight: {pos_weight_val:.2f}")

    train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    eval_criterion  = nn.BCEWithLogitsLoss()

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

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # --- Training loop ----------------------------------------------------
    best_f1 = 0.0
    patience = 10
    epochs_no_improve = 0
    history = {"train": [], "val": []}
    

    for epoch in range(1, args.epochs + 1):
        train_m, _, _ = run_epoch(model, train_loader, train_criterion, device, optimizer)
        val_m, _, _   = run_epoch(model, val_loader,   eval_criterion,  device)
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
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} - no val F1 imporvement for {patience} epochs")
                break
                
    # --- Test evaluation --------------------------------------------------
    test_m, test_labels, test_probs = run_epoch(model, test_loader, eval_criterion, device)
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
    #--- Metrics output --------------------------------------------------
    
    plot_calibration(test_labels, test_probs, plot_dir)
    roc_auc, pr_auc = plot_roc_and_pr(test_labels, test_probs, plot_dir)
    
    with open(output_dir / "test_results.txt", "w") as f:
        f.write("=== Test Results ===\n")
        f.write(f"Accuracy  : {test_m['accuracy']:.4f}\n")
        f.write(f"Precision : {test_m['precision']:.4f}\n")
        f.write(f"Recall    : {test_m['recall']:.4f}\n")
        f.write(f"F1        : {test_m['f1']:.4f}\n")
        f.write(f"ROC AUC   : {roc_auc:.4f}\n")
        f.write(f"PR AUC    : {pr_auc:.4f}\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  TN: {cm[0,0]}  FP: {cm[0,1]}\n")
        f.write(f"  FN: {cm[1,0]}  TP: {cm[1,1]}\n")
        f.write(f"\nTraining args: {vars(args)}\n")
    print("Saved test results to:", output_dir / "test_results.txt")
    
    cm = confusion_matrix(test_labels, (test_probs > 0.3).astype(int))
    print("\nConfusion matrix (test set):")
    print(f"  True  negatives : {cm[0,0]:>6}   False positives : {cm[0,1]:>6}")
    print(f"  False negatives : {cm[1,0]:>6}   True  positives : {cm[1,1]:>6}")
    print(f"\n  Of {cm[1].sum()} actual crossings, model found {cm[1,1]} ({cm[1,1]/cm[1].sum()*100:.1f}%    )")
    print(f"  Of {cm[0].sum()} actual backgrounds, model flagged {cm[0,1]} as crossings ({cm[0,1]/cm[0]     .sum()*100:.1f}%)")

if __name__ == "__main__":
    main()