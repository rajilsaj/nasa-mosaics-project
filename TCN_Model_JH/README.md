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
├── data/
│   ├── processed/2004/        # Raw .nc files from Cassini ELS instrument
│   ├── new_processed/         # Labelled .nc files (output of add_crossing_event_new.py)
│   └── zenodo-3946033/
│       └── crossings/labels/  # YAML files with expert-annotated crossing timestamps
│           ├── bs/all/        # Bow shock crossings  → label 1
│           ├── mp/all/        # Magnetopause crossings → label 1
│           ├── dg/            # Data gaps  → files excluded from training
│           └── sc/            # Spacecraft manoeuvres → files excluded from training
│
├── add_crossing_event_new.py  # Step 1: attach labels to .nc files
├── data_loader.py             # Step 2: PyTorch dataset and DataLoader
├── train.py                   # Step 3: train the TCN model
├── inference.py               # Step 4: run predictions on new files
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add crossing labels to raw .nc files
This reads the Zenodo YAML labels and writes a `crossing_event` variable into each `.nc` file. Files flagged as data gaps (`dg/`) or spacecraft manoeuvres (`sc/`) are skipped entirely.
```bash
python add_crossing_event_new.py
```
Output goes to `data/new_processed/`.

### 3. Train the model
```bash
python train.py --data-dir data/new_processed
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

Raw `.nc` files → label attachment → sliding windows → train/val/test split → model training.

The full pipeline is documented in `Data_Preparation_Roadmap.docx`. Key decisions:

| Decision | Choice | Why |
|---|---|---|
| Label source | `bs/all/` and `mp/all/` | Both crossing directions combined — more training samples |
| Exclusions | `dg/` and `sc/` files | Corrupt data would teach the model wrong patterns |
| Preprocessing | log10 then z-score normalisation | ELS counts span many orders of magnitude |
| Window size | 128 timesteps (~17 min) | Long enough to capture approach + crossing + departure |
| Window label | Centre timestep | Cleanest approach for change-point detection |
| Imbalance fix | Undersample majority class + class weights | Preserves temporal structure; no synthetic data |
| Train/val/test split | By day of year | Prevents leakage across orbit crossings |

**Split:** Train = days 1–270 (Jan–Sep) · Val = days 271–335 (Oct–Nov) · Test = days 336–366 (Dec)

---

## Training Options

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/new_processed` | Path to labelled `.nc` files |
| `--sequence-length` | `128` | Timesteps per window |
| `--stride` | `16` | Window slide step |
| `--batch-size` | `32` | Sequences per training batch |
| `--epochs` | `50` | Training epochs |
| `--lr` | `1e-3` | Learning rate |
| `--dropout` | `0.2` | Dropout regularisation |
| `--num-channels` | `64 128 256 128` | TCN channel sizes per layer |
| `--max-files` | None | Cap number of files (useful for quick tests) |

---

## Model Output

The model outputs a crossing probability (0–1) for each timestep. A timestep is classified as a crossing when the probability exceeds 0.5 (adjustable with `--threshold`).

Training is evaluated using **F1 score** — this is more informative than accuracy when crossings are rare, because a model that always predicts "no crossing" would have high accuracy but zero F1.

---

## Scaling to 2004–2012

The pipeline is designed to scale. When the full multi-year dataset is available, only one change is needed — replace the day-of-year split in `data_loader.py` with a year-based split:

| Split | Years |
|---|---|
| Train | 2004–2009 |
| Val | 2010–2011 |
| Test | 2012 |

All other code remains unchanged.

---

## Requirements

Python 3.10+ · See `requirements.txt` for full list. Core dependencies: `torch`, `xarray`, `numpy`, `scikit-learn`, `matplotlib`.
