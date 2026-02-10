# Comprehensive Analysis Workspace Structure

## Directory Organization

All work is contained within `comprehensive_analysis/` folder.

```
comprehensive_analysis/
├── README.md                          # Workspace overview
├── APPROACH_OUTLINE.md                # Overall approach
├── EXPERT_RF_STRATEGY.md              # RF expert strategy
├── class_prior_integration_strategy.md # Class prior methods
├── WORKSPACE_STRUCTURE.md             # This file
│
├── Scripts/
│   ├── analyze_comprehensive_dataset.py
│   ├── data_preparation.py            # Temporal splitting & window extraction
│   ├── feature_engineering.py         # Feature creation (15 + autoencoder)
│   ├── train_rf_model.py              # Model training
│   ├── evaluate_model.py              # Model evaluation
│   ├── class_prior_adjustment.py     # Probability adjustment
│   └── ...
│
├── data/
│   ├── splits/                        # Temporal splits
│   │   ├── ml_train.csv
│   │   ├── ml_val.csv
│   │   ├── ml_test.csv
│   │   ├── jackson_train.csv
│   │   ├── jackson_val.csv
│   │   └── jackson_test.csv
│   │
│   ├── windows/                       # Extracted windows
│   │   ├── train_windows.csv
│   │   ├── val_windows.csv
│   │   └── test_windows.csv
│   │
│   └── features/                     # Engineered features
│       ├── train_features.csv
│       ├── val_features.csv
│       └── test_features.csv
│
├── models/                            # Trained models
│   ├── rf_comprehensive_*.pkl
│   └── model_metadata_*.json
│
└── results/                          # All outputs
    ├── plots/                        # Visualizations
    ├── reports/                      # Text reports
    └── metrics/                     # JSON/metrics files
```

## File Path Conventions

- **Input data**: Reference parent directory with `../`
  - `../comprehensive_filtered_data_optimized.csv`
  - `../Jackson_vortex_detections_reformatted_augmented.csv`

- **Output data**: Save in `comprehensive_analysis/data/`
- **Scripts**: Save in `comprehensive_analysis/` root
- **Results**: Save in `comprehensive_analysis/results/`
- **Models**: Save in `comprehensive_analysis/models/`

## Important Notes

✅ All scripts run from `comprehensive_analysis/` directory
✅ All outputs stay within `comprehensive_analysis/`
✅ Reference parent files with `../` prefix
✅ Never create files outside this folder

