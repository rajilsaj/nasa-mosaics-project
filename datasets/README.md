# Datasets — Mars Vortex Detection

Central home for every dataset in the project: the originals (before the
split) and everything derived from them. The temporal split is done ONCE
here and shared by both model pipelines (`rf/` and `xgboost/`), so the two
models always train and evaluate on exactly the same data.

## Layout

```
datasets/
├── Jackson_vortex_detections_reformatted_augmented.csv   ORIGINAL - labeled vortex events (294)
├── ml_ready_vortex_data.csv                              ORIGINAL - raw pressure stream (MISSING, see below)
│
├── temporal_splits/                    STEP 1 - raw data cut by time (70/15/15 + gaps)
│   ├── jackson_train.csv / jackson_val.csv / jackson_test.csv
│   ├── ml_train.csv (MISSING, see below) / ml_val.csv / ml_test.csv
│   └── split_summary.txt
│
├── train_windows.csv / val_windows.csv / test_windows.csv        STEP 2 - extracted windows
├── train_balanced.csv / val_balanced.csv                         STEP 2b - class-balanced variants
│
├── train_features.csv / val_features.csv / test_features.csv     STEP 3 - 15 engineered features
│                                                                 (what rf/ and xgboost/ train on)
└── val_sliding_features.csv / test_sliding_features.csv          STEP 4 - continuous sliding-window
                                                                  stream (deployment evaluation)
```

## Who reads what

| Consumer | Files used |
|---|---|
| `rf/` and `xgboost/` training | `train_features.csv`, `val_features.csv`, `test_features.csv` |
| `xgboost/evaluate_sliding_xgb.py` | `val_sliding_features.csv`, `test_sliding_features.csv` |
| `core pipeline scripts/` (regeneration) | originals -> `temporal_splits/` -> windows -> features |
| Root analysis scripts | various of the above (paths updated to `datasets/...`) |

## Missing files

Two files are too large for GitHub and were removed from tracking; they are
expected here if ever recovered (from the machine where the pipeline was
originally run, or by regenerating from the raw mission data):

- `ml_ready_vortex_data.csv` (~3.3M samples) — the full unsplit dataset
- `temporal_splits/ml_train.csv` (~2.5M samples) — its training portion

Training the models does NOT need them (the feature files above are
complete); they are only required to redo the split or re-extract windows
from scratch.

## Note on legacy folders

`old/` (archived experiments) and `comprehensive_analysis/` (self-contained
workspace with its own `data/` layout) intentionally keep their own paths
and do not read from this folder.
