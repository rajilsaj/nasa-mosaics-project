# Two-Stage LSTM Vortex Detection Pipeline

A power-efficient two-stage LSTM pipeline for detecting Martian dust vortices using pressure sensor data.

## Overview

This pipeline uses a **two-stage approach** to balance vortex detection accuracy with power efficiency:

1. **Stage 1: LSTM Autoencoder** - Filters out normal negative data, keeping only artifacts and possible vortex data
2. **Stage 2: LSTM Classifier** - Detects vortices in the filtered data

## Architecture

### Stage 1: Autoencoder Filter
- **Input**: Detrended raw pressure data (single feature)
- **Architecture**: LSTM encoder-decoder (32 units)
- **Purpose**: Anomaly detection to filter out normal data
- **Output**: Reconstruction error threshold for filtering

### Stage 2: Classifier
- **Input**: Detrended raw pressure data (single feature)
- **Architecture**: LSTM classifier (32 units)
- **Purpose**: Binary classification (vortex vs. normal)
- **Output**: Probability of vortex detection

## Performance

- **Event-based F1-Score**: 0.5520
- **Event Recall**: 0.6749 (on filtered data)
- **Event Precision**: 0.4670
- **Overall Pipeline Recall**: 45.2% (considering autoencoder filtering)

## Files

- `lstm_autoencoder.py` - Stage 1 autoencoder training and evaluation
- `classifier_model.py` - Stage 2 classifier training and evaluation
- `prepare_filtered_data.py` - Helper script to prepare filtered data for classifier
- `event_based_evaluation.py` - Event-based evaluation with latch-on logic
- `two_stage_pipeline.py` - Orchestrates both stages for inference

## Quick Start

### 1. Train Autoencoder
```bash
python lstm_autoencoder.py --data_path ../../../data/ml_ready_vortex_data.csv
```

### 2. Prepare Filtered Data
```bash
python prepare_filtered_data.py --data_path ../../../data/ml_ready_vortex_data.csv --threshold 85
```

### 3. Train Classifier
```bash
python classifier_model.py --filtered_data filtered_data_85.pkl --data_path ../../../data/ml_ready_vortex_data.csv
```

### 4. Evaluate Classifier Only (Using Pretrained Model)
```bash
python classifier_model.py --filtered_data filtered_data_85.pkl.gz --data_path [path_to_ml_ready_vortex_data.csv]
```

### 5. Evaluate Full Pipeline (Not implemented yet)
```bash
python two_stage_pipeline.py --data_path ../../../data/ml_ready_vortex_data.csv 
```

## Model Files

- `autoencoder_model.h5` - Trained autoencoder (included)
- `classifier_model.h5` - Trained classifier (included)
- `filtered_data_85.pkl.gz` - Compressed filtered dataset for classifier training (available via Google Drive)

**Note**: Pretrained models are included in this repository. The large filtered data file is available via Google Drive link.

## Event-Based Evaluation

The pipeline uses a **latch-on mechanism** for realistic evaluation:

1. **Single detection** during `gt_detection_win` triggers the entire event
2. **All subsequent points** in the event are counted as TP until `gt_fwhm` ends
3. **Event-level success** rather than point-wise accuracy

## Results Summary

| Metric | Value |
|--------|-------|
| Overall Pipeline Recall | 45.2% |
| Point-wise F1-Score | 0.4739
| Event-based F1-Score | 0.5520 |
| Power Savings | 83.4% |
| Energy Efficiency Ratio | 1.8x savings per vortex |

This pipeline provides an excellent balance between vortex detection accuracy and power efficiency for Martian exploration missions. 