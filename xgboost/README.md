# XGBoost — Mars Vortex Detection

XGBoost (eXtreme Gradient Boosting) implementation of the vortex detector,
built as a drop-in counterpart to the Random Forest pipeline in
`core pipeline scripts/train_rf_model.py`. It uses the **same datasets, same
15 features, and same evaluation report format**, so the two models can be
compared line by line.

## Files

| File | Purpose |
|---|---|
| `run.py` | **Entry point** — checks environment and data, then trains and produces results |
| `requirements.txt` | Packages needed by this folder (for creating a dedicated virtual env) |
| `train_xgb_model.py` | Trains the XGBoost classifier, tunes the decision threshold on validation, evaluates on test, saves model + threshold |
| `tune_xgb_hyperparams.py` | Grid search over key hyperparameters; its winner is applied automatically by `run.py --tune` |
| `evaluate_sliding_xgb.py` | Deployment-realistic evaluation on the continuous sliding-window stream (natural <1% imbalance, FP/hour) |
| `visualize_model.py` | Generates PNGs of the trained model's structure (ensemble overview + one boosted tree as a flowchart, no Graphviz needed) |
| `models/` | Created on first run — saved models (`.json` + `.pkl`) and metadata (including the tuned threshold) |
| `results/` | Created on first run — feature importance, threshold sweep, sliding-window results, tuning results |

## Quick start

From **inside the xgboost folder** (use `py` if `python` is not on your PATH):

```
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python run.py
```

Other commands:

```
python run.py --check    # verify dependencies + data files only, run nothing
python run.py --tune     # hyperparameter search -> best params applied -> train
```

`run.py` chains the full pipeline: (optional) hyperparameter search,
training with threshold tuning, sliding-window evaluation, and finally the
model-structure PNGs (`xgb_ensemble_overview.png`, `xgb_tree_0_structure.png`
in `results/`). The individual scripts can also be run directly
(`python train_xgb_model.py`, `python evaluate_sliding_xgb.py`,
`python visualize_model.py --tree N` — the latter two load the latest saved
model).

## Why the extra evaluation stages

- **Decision threshold tuning** — `predict()`'s default 0.5 cutoff is
  arbitrary here: `scale_pos_weight` skews probabilities on purpose, and the
  project's RF work found its F1-optimal threshold near 0.90. Training sweeps
  all thresholds on the validation set, picks the F1-optimal one, reports
  high-precision / high-recall alternatives, and stores the chosen value in
  the model metadata (`decision_threshold`).
- **Sliding-window evaluation** — window-level metrics (~87% F1 for RF) are
  measured on curated, balanced windows. On the continuous sliding stream with
  natural <1% vortex rate, RF collapsed to ~9% F1. XGBoost is therefore scored
  on `../datasets/val_sliding_features.csv` and
  `../datasets/test_sliding_features.csv`,
  reporting precision/recall/F1, PR-AUC, and **false positives per hour**
  (from real SCLK timestamps) at both the tuned and default thresholds.
  Results land in `results/xgb_sliding_window_results.csv` — compare against the RF baseline
  in `../threshold_calibration_results.csv`.

## What XGBoost adds over the Random Forest

| Capability | Random Forest (current) | XGBoost (this folder) |
|---|---|---|
| Tree building | Independent trees, majority vote | Sequential trees, each corrects the previous ones' errors |
| Class imbalance | `class_weight='balanced'` | `scale_pos_weight = n_negative / n_positive` (computed from the data) |
| Number of trees | Fixed (100) | Up to 1000, chosen automatically by **early stopping** on the validation set |
| Regularization | Only depth/leaf limits | Built-in L2 (`reg_lambda`), L1 (`reg_alpha`), and `gamma` split pruning |
| Missing values | Must be imputed first | Handled natively (learns best direction per split) |
| Evaluation during training | None | Watches validation PR-AUC (`aucpr`) every round — suited to the rare-vortex imbalance |
| Feature importance | Impurity-based | **Gain-based** (average loss reduction per feature), plus split counts |
| Deployment format | Pickle only | Native portable `.json` (loadable from C/C++ runtimes — good for Snapdragon-class on-board deployment) + `.pkl` for Python parity |

## Data used (unchanged from the RF pipeline)

All datasets live in the shared `../datasets/` folder (see its README):

- `../datasets/train_features.csv` — training split
- `../datasets/val_features.csv` — validation split (drives early stopping + threshold-independent metrics)
- `../datasets/test_features.csv` — held-out test split (final evaluation only)

The temporal 3-way split with gaps is preserved. The tuning script deliberately
does **not** use k-fold cross-validation: shuffling time-ordered windows across
folds would leak future data into training.

## Outputs

After training, `models/` contains:

- `xgb_vortex_detector_<timestamp>.json` — native XGBoost model (deployment)
- `xgb_vortex_detector_<timestamp>.pkl` — joblib pickle (Python evaluation scripts)
- `model_metadata_<timestamp>.json` — hyperparameters, best iteration, validation metrics

and `results/` contains `xgb_feature_importance.csv` (gain + split counts per feature).

Note: like the RF models, `.pkl` files are excluded by the root `.gitignore`;
the native `.json` model **is** committable.

## ⚠️ Folder-name caveat

This folder is named `xgboost`, the same as the pip package. Running the
scripts directly (`python xgboost\train_xgb_model.py`) is safe. However, do
**not** open a Python session or notebook *inside this folder's parent with
this folder on `sys.path`* and expect `import xgboost` to work oddly — if you
ever see `AttributeError: module 'xgboost' has no attribute 'XGBClassifier'`,
Python picked up this folder instead of the installed package. Renaming the
folder to e.g. `xgboost_model` removes the risk entirely.
