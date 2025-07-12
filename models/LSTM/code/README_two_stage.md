# Two-Stage LSTM Pipeline for Vortex Detection

This implementation provides a two-stage LSTM approach for vortex detection that combines the benefits of autoencoder filtering with LSTM classification.

## Overview

The pipeline consists of two stages:

### Stage 1: LSTM Autoencoder Filter
- **Purpose**: Filters out non-artifact/positive vortexes
- **Training**: Trained on normal (non-vortex, non-artifact) pressure patterns
- **Output**: Reconstruction error threshold to identify potential vortex events
- **Function**: Acts as a pre-filter to reduce false positives

### Stage 2: LSTM Classifier
- **Purpose**: Distinguishes between artifacts and actual vortexes
- **Training**: Trained on sequences that pass the autoencoder filter
- **Output**: Binary classification (vortex vs artifact)
- **Function**: Fine-grained classification of filtered sequences

### Inference Pipeline
1. Input sequence goes through autoencoder
2. Sequences with high reconstruction error (potential vortexes) are kept
3. Filtered sequences are classified by the second LSTM
4. Final prediction combines both stages

## Files

- `two_stage_lstm.py`: Main implementation of the two-stage pipeline
- `run_two_stage_pipeline.py`: Simple demo script
- `lstm_autoencoder.py`: Original autoencoder implementation (for reference)
- `lstm_model.py`: Original single-stage LSTM implementation (for reference)

## Usage

### Basic Usage

```python
from two_stage_lstm import TwoStageLSTMPipeline

# Initialize pipeline
pipeline = TwoStageLSTMPipeline(window_size=60, debug=True)

# Load and prepare data
data = pd.read_csv('your_data.csv')
sequences, labels = pipeline.prepare_sequences(data)

# Train autoencoder on normal data
normal_sequences = pipeline.prepare_normal_sequences_for_autoencoder(data)
pipeline.train_autoencoder(normal_sequences, epochs=50)
pipeline.set_autoencoder_threshold(normal_sequences)

# Filter and train classifier
filtered_X_train, filtered_indices = pipeline.filter_with_autoencoder(X_train)
filtered_y_train = y_train[filtered_indices]
pipeline.train_classifier(filtered_X_train, filtered_y_train, X_val, y_val)

# Make predictions
predictions, filtered_indices = pipeline.predict(X_test)
```

### Command Line Usage

```bash
# Train the pipeline
python two_stage_lstm.py --data_path your_data.csv --train --window_size 60

# Evaluate the pipeline
python two_stage_lstm.py --data_path your_data.csv --evaluate --window_size 60

# Both train and evaluate
python two_stage_lstm.py --data_path your_data.csv --train --evaluate --debug
```

### Demo Script

```bash
# Update the data_path in run_two_stage_pipeline.py first
python run_two_stage_pipeline.py
```

## Key Features

### Autoencoder Stage
- **Input**: Detrended raw pressure sequences
- **Architecture**: LSTM encoder-decoder with reconstruction loss
- **Training**: Only on normal (non-vortex, non-artifact) sequences
- **Threshold**: 95th percentile of reconstruction error on normal data
- **Output**: Filtered sequences with high reconstruction error

### Classifier Stage
- **Input**: Sequences that pass autoencoder filter
- **Architecture**: Bidirectional LSTM with dense layers
- **Training**: Binary classification on filtered sequences
- **Loss**: Binary crossentropy with class balancing
- **Output**: Probability of vortex vs artifact

### Data Processing
- **Detrending**: Local mean subtraction for each window
- **Window Size**: Configurable (default: 60)
- **Artifact Filtering**: Excludes known artifacts from training
- **Balanced Sampling**: Creates balanced dataset for training

## Advantages

1. **Reduced False Positives**: Autoencoder pre-filter removes obvious non-vortex patterns
2. **Focused Classification**: Second LSTM only sees potential vortex sequences
3. **Artifact Handling**: Explicit artifact detection and filtering
4. **Interpretable**: Two-stage approach provides interpretable intermediate results
5. **Flexible**: Each stage can be tuned independently

## Performance Metrics

The pipeline provides comprehensive evaluation:
- **Precision/Recall/F1**: Standard classification metrics
- **AUC/AP**: Area under ROC curve and average precision
- **Filter Ratio**: Percentage of sequences passing autoencoder filter
- **Confusion Matrix**: Detailed breakdown of predictions

## Model Files

- `autoencoder_model.h5`: Trained autoencoder model
- `classifier_model.h5`: Trained classifier model
- `best_autoencoder.h5`: Best autoencoder during training
- `best_classifier.h5`: Best classifier during training

## Configuration

Key parameters:
- `window_size`: Sequence length (default: 60)
- `autoencoder_epochs`: Training epochs for autoencoder (default: 50)
- `classifier_epochs`: Training epochs for classifier (default: 50)
- `batch_size`: Batch size for training (default: 256)
- `learning_rate`: Learning rate (default: 0.001)
- `reconstruction_threshold_percentile`: Autoencoder threshold (default: 95)

## Comparison with Single-Stage Approach

| Aspect | Single-Stage LSTM | Two-Stage Pipeline |
|--------|------------------|-------------------|
| **Complexity** | Simple | More complex |
| **False Positives** | Higher | Lower (filtered) |
| **Training Data** | All sequences | Filtered sequences |
| **Interpretability** | Low | High (two stages) |
| **Computational Cost** | Lower | Higher |
| **Artifact Handling** | Implicit | Explicit |

## Future Improvements

1. **Ensemble Methods**: Combine multiple autoencoders or classifiers
2. **Attention Mechanisms**: Add attention to both stages
3. **Online Learning**: Adapt thresholds during deployment
4. **Multi-class Classification**: Distinguish vortex types
5. **Temporal Consistency**: Add temporal smoothing to predictions 