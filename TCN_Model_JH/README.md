# Magnetic Field Boundary Detection — TCN Model

Detects bow shock and magnetopause crossings in Cassini spacecraft plasma data using a Temporal Convolutional Network (TCN). The model reads time-series energy spectrogram data from the ELS instrument and predicts, per timestep, whether a magnetic boundary crossing is occurring.

---

## What This Project Does

The Cassini spacecraft recorded plasma measurements as it crossed Saturn's magnetic boundaries. This project trains a deep learning model to automatically detect those crossings, which would otherwise require manual expert labelling.

**Two boundaries are detected:**
- **Bow shock (BS)** — where the solar wind first hits Saturn's magnetic field
- **Magnetopause (MP)** — the inner boundary of Saturn's magnetosphere

The model outputs a binary prediction per timestep: `0 = no crossing`, `1 = crossing`.

---

## Project Structure

```
project/
│
├── dataset_index/
│   ├── dataset_index_2004.csv     # Master index of all ELS files with split + label info
│   ├── preprocess/                # Per-file log-scaled arrays (_t_ns.npy, _X_63.npy)
│   ├── processed/labels_2004/    # Per-file label arrays (_y.npy)
│   └── windows_2004/
│       └── scaled/                # Final train/val/test window arrays ← model reads these
│           ├── train_X.npy        # shape (N, 128, 63)  float32
│           ├── train_y.npy        # shape (N,)          uint8
│           ├── val_X.npy
│           ├── val_y.npy
│           ├── test_X.npy
│           └── test_y.npy
│
├── data/
│   └── zenodo-3946033/
│       └── crossings/labels/      # YAML files with expert-annotated crossing timestamps
│           ├── bs/all/            # Bow shock crossings  → label 1
│           ├── mp/all/            # Magnetopause crossings → label 1
│           ├── dg/                # Data gaps  → files excluded from training
│           └── sc/                # Spacecraft manoeuvres → files excluded from training
│
├── dataset_index_builder.py   # Step 1: build the master CSV index
├── preprocess_from_masterindex.py  # Step 2: log-scale raw CSVs → .npy arrays
├── labelbuilder_2004.py       # Step 3: attach crossing labels → _y.npy files
├── windowbuilder.py           # Step 4: cut windows + train/val/test split → windows_2004/
├── fit_scaler.py              # Step 5: z-score normalise windows → windows_2004/scaled/
├── class_weightbuilder.py     # Step 6: compute class weights for loss function
│
├── data_loader.py             # Loads pre-built .npy windows for training
├── train.py                   # Train the TCN model
├── tcn_model.py               # TCN architecture
├── inference.py               # Run predictions on new files
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the data pipeline
Run these scripts in order. Each step reads from the previous step's output.

```bash
# Step 1 — build the master file index (CSV with split + label assignments)
python dataset_index_builder.py

# Step 2 — log-scale raw CSV files into per-file .npy arrays
python preprocess_from_masterindex.py

# Step 3 — generate per-timestep labels from the Zenodo YAML crossing annotations
python labelbuilder_2004.py

# Step 4 — cut sliding windows and split into train / val / test
python windowbuilder.py

# Step 5 — z-score normalise the windows (fit on train, apply to all)
python fit_scaler.py

# Step 6 — compute class weights (used by train.py for the loss function)
python class_weightbuilder.py
```

Output goes to `dataset_index/windows_2004/scaled/`.

### 3. Train the model
```bash
python train.py --data-dir dataset_index/windows_2004/scaled
```
The best checkpoint (by validation F1) is saved to `checkpoints/best_model.pt`.

### 4. Run inference on new files
```bash
# Single file
python inference.py --model checkpoints/best_model.pt --input path/to/file.nc

# Whole folder — saves a prediction plot per file
python inference.py --model checkpoints/best_model.pt --input path/to/folder/ --output plots/
```

---

## Data Pipeline

Raw CSV files → log-scaling → label attachment → adaptive sliding windows → z-score normalisation → model training.

Key decisions:

| Decision | Choice | Why |
|---|---|---|
| Label source | `bs/all/` and `mp/all/` | Both crossing directions combined — more training samples |
| Exclusions | `dg/` and `sc/` files | Corrupt data would teach the model wrong patterns |
| Anode reduction | Sum across 8 anodes | Collapses (time, 63, 8) → (time, 63) while preserving total counts |
| Preprocessing | log10 then z-score normalisation | ELS counts span many orders of magnitude |
| Window size | 128 timesteps (~17 min) | Long enough to capture approach + crossing + departure |
| Window label | Centre timestep | Cleanest approach for change-point detection |
| Imbalance fix | Adaptive stride + pos_weight | Dense sampling near crossings, sparse sampling elsewhere |
| Train/val/test split | By day of year | Prevents leakage across orbit crossings |

**Split:** Train = days 1–270 (Jan–Sep) · Val = days 271–335 (Oct–Nov) · Test = days 336–366 (Dec)

---

## Adaptive Sliding Window

Crossing events are rare — without special handling the model sees thousands of
background windows for every one crossing window and learns to always predict
"no crossing". The adaptive sliding window in `windowbuilder.py` fixes this without generating fake data.

### How it works

For each file, a single pass builds windows with a stride that changes depending on proximity to a crossing:

- **Near a crossing** (within `NEAR_RADIUS` timesteps): small stride (`STRIDE_NEAR = 4`, ~32 s) — dense overlapping windows so the model sees the event from many angles
- **Away from a crossing**: large stride (`STRIDE_FAR = 16`, ~2 min) — sparse sampling to keep background windows manageable

Each window is labelled by its own centre timestep: `1` if a crossing is occurring at that point, `0` otherwise.

```
File timeline:
──────────────────[===crossing===]──────────────────
                ↑               ↑
         centre - NEAR_RADIUS   centre + NEAR_RADIUS

Dense windows (STRIDE_NEAR):        Sparse windows (STRIDE_FAR):
  [w1] [w2] [w3] [w4] [w5] ...      [w ]    [w ]    [w ]  ...
```

### Checking your class balance

After `windowbuilder.py` finishes it prints a balance summary per split:

```
TRAIN:
  windows: 52400
  positives: 3100
  positive fraction: 0.059
```

After `data_loader.py` loads the data it also reports:

```
  Dataset built — background: 49300, crossing: 3100 (ratio 15.9:1)
```

If the ratio is very high (above ~50:1) try reducing `STRIDE_FAR` or increasing `NEAR_RADIUS` in `windowbuilder.py` to bring more crossing-adjacent windows into the training set.

---

## Training Options

| Flag | Required | Default | Description |
|---|---|---|---|
| `--data-dir` | ✅ yes | — | Path to folder containing the six `.npy` window files |
| `--batch-size` | no | `32` | Sequences per training batch |
| `--epochs` | no | `50` | Training epochs |
| `--lr` | no | `1e-3` | Learning rate |
| `--dropout` | no | `0.2` | Dropout regularisation |
| `--num-channels` | no | `64 128 256 128` | TCN channel sizes per layer |
| `--device` | no | `auto` | `cpu`, `cuda`, or `auto` |

Example:
```bash
python train.py --data-dir dataset_index/windows_2004/scaled --epochs 30 --batch-size 64
```

---

## Model Output

The model outputs a crossing probability (0–1) for each timestep. A timestep is classified as a crossing when the probability exceeds 0.5 (adjustable with `--threshold` in `inference.py`).

Training is evaluated using **F1 score** — this is more informative than accuracy when crossings are rare, because a model that always predicts "no crossing" would have high accuracy but zero F1.

---

## Scaling to 2004–2012

The pipeline is designed to scale. When the full multi-year dataset is available, two changes are needed:

**1.** Update the paths in each preprocessing script to point at the full dataset.

**2.** Replace the day-of-year split in `dataset_index_builder.py` with a year-based split:

| Split | Years |
|---|---|
| Train | 2004–2009 |
| Val | 2010–2011 |
| Test | 2012 |

All model and training code remains unchanged.

---

## Requirements

Python 3.10+ · See `requirements.txt` for full list. Core dependencies: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`.
