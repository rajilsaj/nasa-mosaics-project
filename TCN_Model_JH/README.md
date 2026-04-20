# Magnetic Field Boundary Detection — TCN Model

A deep learning model that detects bow shock and magnetopause crossings in Cassini spacecraft plasma data. It reads time-series energy spectrogram data and predicts, per timestep, whether a magnetic boundary crossing is occurring.

---

## What the Model Does

The Cassini spacecraft recorded plasma measurements as it crossed Saturn's magnetic boundaries. This model automatically detects those crossings from the ELS instrument's energy spectrogram data.

**Two boundaries are detected:**
- **Bow shock (BS)** — where the solar wind first hits Saturn's magnetic field
- **Magnetopause (MP)** — the inner boundary of Saturn's magnetosphere

The model outputs a **binary prediction per timestep**: `0 = no crossing`, `1 = crossing`.

---

## What the Model Receives

Each input to the model is a **window** — a short clip of the spectrogram cut from a longer recording.

| Property | Value | Meaning |
|---|---|---|
| Window length | 128 timesteps | ~17 minutes of plasma data |
| Features per timestep | 63 | One value per ELS energy bin |
| Batch shape | `(N, 128, 63)` | N windows at once during training |
| Labels shape | `(N,)` | One label (0 or 1) per window |
| Label source | End timestep | The last timestamp of the window decides the label |

The data arrives **already preprocessed**: raw counts have been `log10` compressed (they span many orders of magnitude) and z-score normalised (each energy bin shifted to mean 0, std 1). The model does not normalise anything itself.

---

## Model Architecture — `tcn_model.py`

The model is a **Temporal Convolutional Network (TCN)**. TCNs process the whole sequence in parallel and use **dilated convolutions** to look further back in time without needing a huge number of parameters.

### Data flow

```
Input: (batch, 128 timesteps, 63 energy bins)
    ↓  transpose — Conv1d expects channels first
(batch, 63, 128)
    ↓  TemporalBlock × 4
(batch, last_channel_size, 128)
    ↓  transpose back
(batch, 128, last_channel_size)
    ↓  Dropout → Linear(last_channel_size → 1)
Output: (batch, 128, 1)  ← one raw logit per timestep
```

Pass through `sigmoid` for probabilities (0–1). A timestep is a crossing when probability > 0.5.

---

### Building block — `TemporalBlock`

Each block applies two **dilated causal convolutions** plus a **residual (skip) connection**.

- **Causal** — the output at time `t` only uses inputs from `t` and earlier. Enforced by left-padding and trimming the right side (`Chomp1d`).
- **Dilated** — the convolution skips timesteps by a dilation factor. With kernel size 3 and dilation 4, it touches `t`, `t-4`, `t-8`. This is how later layers build a long receptive field cheaply.
- **Residual connection** — adds the block's input directly to its output. Keeps gradients healthy during training.

```
Input ──→ Conv1d → Chomp → ReLU → Dropout
       │  Conv1d → Chomp → ReLU → Dropout ──→ + ──→ ReLU ──→ Output
       └─────── residual (1×1 conv or identity) ────────┘
```

---

### The stack — `BowShockTCN`

Four `TemporalBlock` layers are stacked. Each doubles the dilation, so deeper layers see further back in time.

| Layer | Dilation | Approximate look-back (at 8 s/step) |
|---|---|---|
| 0 | 1 | ~24 s |
| 1 | 2 | ~1 min |
| 2 | 4 | ~2 min |
| 3 | 8 | ~4 min |

Default channels: `[64, 128, 256, 128]` — widens then narrows, compressing the representation before making a prediction.

**Key constructor arguments** (`tcn_model.py`, class `BowShockTCN`):

```python
BowShockTCN(
    num_energy_bins = 63,                    # must match input features
    num_channels    = [64, 128, 256, 128],   # one entry per layer
    kernel_size     = 3,
    dropout         = 0.2,
)
```

To make it **smaller** (reduces overfitting on small datasets):
```python
num_channels = [64, 128, 128, 64]
```

To add **more layers**:
```python
num_channels = [64, 128, 256, 256, 128, 64]
```

---

## Training — `train.py`

### Loss function

Uses **`BCEWithLogitsLoss`** (Binary Cross-Entropy with built-in sigmoid). A `pos_weight` penalises missed crossings more heavily than false alarms, compensating for class imbalance.

```python
# train.py — where pos_weight is applied
pos_weight_val = min(class_weights[1], 10.0)  # capped at 10
train_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
```

The cap at 10 stops the model from overcorrecting and predicting crossings everywhere. Raise it if recall is too low.

### Per-window prediction

The model outputs one logit per timestep `(batch, 128, 1)`. These are averaged across time to get one score per window:

```python
# train.py — inside run_epoch()
logits    = model(sequences).squeeze(-1)   # (B, T)
preds_seq = logits.mean(dim=1)             # (B,) — one value per window
```

This single value is compared to the window label (0 or 1) for the loss.

### Evaluation metric — F1 score

Accuracy is not used because always predicting "no crossing" gives high accuracy on an imbalanced dataset. **F1 score** is only high when the model catches most crossings *and* keeps false alarms low.

The best checkpoint saves whenever validation F1 improves:
```python
# train.py — checkpoint condition
if val_m["f1"] > best_f1:
    best_f1 = val_m["f1"]
    torch.save(..., output_dir / "best_model.pt")
```

### Training options

```bash
python train.py --data-dir dataset_index/windows_2004/scaled
```

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | required | Folder with the six `.npy` window files |
| `--batch-size` | `32` | Windows per training step |
| `--epochs` | `50` | Full passes over training data |
| `--lr` | `1e-3` | Learning rate (Adam optimiser) |
| `--dropout` | `0.2` | Fraction of neurons off per step |
| `--num-channels` | `64 128 256 128` | Channel sizes per TCN layer |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--output-dir` | `checkpoints` | Where to save `.pt` files |

---

## Inference — `inference.py`

Inference runs on raw per-file `.npy` arrays (not pre-windowed). A **sliding window** moves across the file and the model scores each window independently. Overlapping windows are averaged at each timestep, smoothing out noise.

```
File timeline ─────────────────────────────────────────────
Window 1:     [════════════]
Window 2:          [════════════]
Window 3:               [════════════]
...
Each timestep's probability = average over all windows that covered it
```

```bash
# Single file
python inference.py \
    --model checkpoints/best_model.pt \
    --input path/to/file_X_63.npy \
    --scaler dataset_index/windows_2004/scaled/scaler.pkl

# Whole folder
python inference.py \
    --model checkpoints/best_model.pt \
    --input path/to/folder/ \
    --scaler dataset_index/windows_2004/scaled/scaler.pkl \
    --output plots/
```

| Flag | Default | Description |
|---|---|---|
| `--model` | required | Path to `.pt` checkpoint |
| `--input` | required | `_X_63.npy` file or folder |
| `--scaler` | required | Path to `scaler.pkl` |
| `--sequence-length` | `100` | Window size in timesteps |
| `--stride` | `10` | Steps between windows (smaller = smoother, slower) |
| `--threshold` | `0.5` | Probability cutoff for a positive prediction |
| `--output` | None | Folder for plots (omit to display interactively) |

Each file produces a two-panel plot: the energy spectrogram on top, crossing probability with predicted events shaded red on the bottom.

---

## Checkpoint Format

Both `best_model.pt` and `final_model.pt` contain:

```python
{
    "epoch":            int,   # epoch saved at
    "model_state_dict": dict,  # weights — load with model.load_state_dict()
    "val_metrics":      dict,  # accuracy, precision, recall, F1 on val set
    "args":             dict,  # training flags used (num_channels, dropout, etc.)
}
```

The saved `args` are used at inference time to rebuild the exact same architecture before loading weights — so you never have to manually track what settings were used.

---

## Requirements

Python 3.10+ · See `requirements.txt`.  
Core: `torch`, `numpy`, `scikit-learn`, `matplotlib`, `pandas`, `joblib`.
