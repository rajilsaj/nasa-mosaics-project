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
| Imbalance fix | Crossing-centred sliding windows + pos_weight | Preserves temporal structure; no synthetic data |
| Train/val/test split | By day of year | Prevents leakage across orbit crossings |

**Split:** Train = days 1–270 (Jan–Sep) · Val = days 271–335 (Oct–Nov) · Test = days 336–366 (Dec)

---

## Crossing-Centred Sliding Window

Crossing events are rare — without special handling the model sees thousands of
background windows for every one crossing window and learns to always predict
"no crossing". The sliding window strategy fixes this without generating fake data.

### How it works

For every file, the data loader makes two passes:

**Pass 1 — Crossing-centred windows**

Each crossing event in the file is located and its midpoint is found. A context
region is then defined extending `context_hours` hours before and after that
midpoint. A sliding window of length `sequence_length` sweeps across this region
with a small step size (`crossing_stride`), producing many overlapping windows.
Each window is labelled by its own centre timestep — windows sitting on the
crossing get label 1, windows in the surrounding context get label 0. This gives
the model dense coverage of what the plasma looks like approaching, during, and
departing each boundary.

```
File timeline:
──────────────────[===crossing===]──────────────────
         ↑                              ↑
  midpoint - context_hours       midpoint + context_hours

Sliding window sweeps this region with step = crossing_stride:
  [window 1   ] label = 0  (pre-crossing context)
    [window 2   ] label = 0
      [window 3   ] label = 1  (centre lands on crossing)
        [window 4   ] label = 1
          [window 5   ] label = 0  (post-crossing context)
            ...
```

**Pass 2 — Background windows**

A standard sliding window sweeps the whole file with a larger step size
(`background_stride`). These windows are almost entirely label 0 and represent
normal solar wind or magnetosphere interior conditions.

### Controlling the window size and density

All three parameters below can be set at the command line when running `train.py`:

| Parameter | Flag | Default | Effect |
|---|---|---|---|
| Context size | `--context-hours` | `24` | Hours of data each side of the crossing midpoint. Increase to give the model more pre/post context; decrease to focus tightly on the event. |
| Crossing stride | `--crossing-stride` | `4` | Timestep advance between windows inside the context region (~32 s at default). Smaller = more crossing windows = better class balance. |
| Background stride | `--stride` | `16` | Timestep advance between background windows (~2 min at default). Larger = fewer background windows = better class balance. |

At 8 s cadence: 1 hour = 450 timesteps, so `--context-hours 24` covers
roughly 10 800 timesteps either side of the crossing midpoint.

### Example commands

Default settings — 24 h context, ~32 s crossing step, ~2 min background step:
```bash
python train.py --data-dir data/new_processed
```

Tighter context around the crossing (6 h each side), denser crossing sampling:
```bash
python train.py --data-dir data/new_processed \
    --context-hours 6 \
    --crossing-stride 2
```

Wide context (48 h each side) to capture long-range plasma changes before the boundary:
```bash
python train.py --data-dir data/new_processed \
    --context-hours 48 \
    --crossing-stride 8 \
    --stride 32
```

Quick test run on a small subset of files:
```bash
python train.py --data-dir data/new_processed \
    --max-files 10 \
    --context-hours 12 \
    --epochs 5
```

### Checking your class balance

After the data loader builds the dataset it prints a balance summary:

```
  Dataset built — background: 48200, crossing: 3100  (ratio 15.5:1)
  Recommended pos_weight for BCEWithLogitsLoss = 15.5
```

If the ratio is very high (above ~50:1) try reducing `--stride` or increasing
`--context-hours` to bring more crossing windows into the training set.

---

## Training Options

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/new_processed` | Path to labelled `.nc` files |
| `--sequence-length` | `128` | Timesteps per window (~17 min) |
| `--stride` | `16` | Background window slide step (~2 min) |
| `--context-hours` | `24` | Hours each side of crossing midpoint for dense sampling |
| `--crossing-stride` | `4` | Crossing window slide step (~32 s) |
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
