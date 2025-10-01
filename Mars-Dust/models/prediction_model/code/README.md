# Vortex Prediction Model

This script trains or evaluates a vortex prediction model using machine learning techniques.

## Command Line Options

### `--data-fraction`
- **Type**: float
- **Default**: 1.0 (100%)
- **Description**: Fraction of data to use for training (range: 0.0 to 1.0). When loading an existing model, this determines how much data to use for evaluation.
- **Example**: `--data-fraction 0.5` (uses 50% of data)

### `--model-path`
- **Type**: string
- **Default**: models/prediction_model/vortex_model.joblib
- **Description**: Path to load existing trained model. If the model doesn't exist at the specified path, a new model will be trained and saved there.
- **Example**: `--model-path models/prediction_model/vortex_model.joblib`

### `--force-recalculate`
- **Type**: flag (no value needed)
- **Default**: False
- **Description**: Force recalculation of features even if they already exist. By default, the script will use cached features if available.
- **Example**: `--force-recalculate`

### `--force-retrain`
- **Type**: flag (no value needed)
- **Default**: False
- **Description**: Force retraining of the model even if it already exists at the specified path. By default, the script will load an existing model if available.
- **Example**: `--force-retrain`

### `--data-path`
- **Type**: string
- **Default**: data/ml_ready_vortex_data.csv
- **Description**: Path to the input data file
- **Example**: `--data-path path/to/your/data.csv`

## Usage Examples

1. Train model with default settings (uses all data and cached features):
```bash
python vortex_prediction_model.py
```

2. Train model using 50% of data:
```bash
python vortex_prediction_model.py --data-fraction 0.5
```

3. Evaluate existing model:
```bash
python vortex_prediction_model.py --model-path models/prediction_model/vortex_model.joblib
```

4. Force feature recalculation:
```bash
python vortex_prediction_model.py --force-recalculate
```

5. Force model retraining:
```bash
python vortex_prediction_model.py --force-retrain
```

6. Force both feature recalculation and model retraining:
```bash
python vortex_prediction_model.py --force-recalculate --force-retrain
```

7. Use custom data file:
```bash
python vortex_prediction_model.py --data-path path/to/your/data.csv
```

## Notes

- The script will automatically create necessary directories for model storage and results
- Results including metrics and visualizations will be saved in the results directory
- The model uses a Random Forest Classifier with balanced class weights
- Feature processing is handled by the FeatureProcessor class
- By default, the script will use cached features if available. Use `--force-recalculate` to force recalculation. 