# Random Forest — Mars Vortex Detection

Self-contained Random Forest pipeline, mirroring the structure of the
`xgboost/` folder. It is an adapted version of
`core pipeline scripts/train_rf_model.py` (same hyperparameters, same
datasets, same 15 features, same evaluation report) with paths fixed so it
runs from inside this folder and keeps its outputs here.

## Files

| File | Purpose |
|---|---|
| `run.py` | **Entry point** — checks environment and data, trains, then generates the model-structure PNGs automatically |
| `requirements.txt` | Packages needed by this folder (for creating a dedicated virtual env) |
| `train_rf_model.py` | Trains the Random Forest, evaluates on validation + test, saves model and metadata |
| `visualize_model.py` | Generates PNGs of the trained model's structure (forest overview + one tree as a flowchart) |
| `models/` | Created on first run — saved model (`.pkl`) and metadata (`.json`) |
| `results/` | Created on first run — feature importance CSV |

## Quick start

From **inside the rf folder**:

Use `py` if `python` is not on your PATH:

```
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
python run.py
```

Other commands:

```
python run.py --check    # verify dependencies + data files only, run nothing
```

## Data used

All datasets live in the shared `../datasets/` folder (see its README):

- `../datasets/train_features.csv` — training split
- `../datasets/val_features.csv` — validation split
- `../datasets/test_features.csv` — held-out test split (final evaluation only)

## Outputs (formats adapted for Random Forest)

After training, `models/` contains:

- `rf_vortex_detector_<timestamp>.pkl` — the trained model in **joblib pickle**
  format, the canonical persistence format for scikit-learn models (Random
  Forest has no native portable format like XGBoost's `.json`)
- `model_metadata_<timestamp>.json` — hyperparameters, feature list,
  training time, and validation metrics

and `results/` contains:

- `rf_feature_importance.csv` — **impurity-based** importances (the Random
  Forest equivalent of XGBoost's gain-based importance)
- `rf_forest_overview.png` + `rf_tree_0_structure.png` — model-structure
  visualizations, generated automatically at the end of `run.py`
  (regenerate with other options via `python visualize_model.py --tree N --depth D`)

Note: `.pkl` files are excluded by the root `.gitignore`, so the trained
model stays local; the metadata `.json` is committable.

## Comparing with XGBoost

Both folders read the same datasets and print identical evaluation reports
(confusion matrix, precision/recall/F1/ROC-AUC on validation and test), so
you can run `rf\run.py` and `xgboost\run.py` and compare the outputs line
by line.
