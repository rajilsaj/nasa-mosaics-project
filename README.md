# Preprocessing Pipeline

This preprocessing pipeline prepares the ELS Cassini crossing dataset for model training. It takes raw indexed data, engineers the feature matrix, builds dense labels, converts the data into training windows, scales the features, and computes class weights for imbalanced learning.

The pipeline is designed so that each script performs one clear stage of preprocessing, and the output of one stage becomes the input to the next. Running the scripts in order is important. (Don't forget to change the file path's to your specific path's.) 

---

## Pipeline Overview

The full preprocessing workflow is:

1. Build the master index  
   `python dataset_index_builder.py`

2. Preprocess into `X_63` + `t_ns`  
   `python preprocess_from_masterindex.py`

3. Build dense `y` labels  
   `python labelbuilder_2004.py`

4. Build window datasets  
   `python windowbuilder.py`

5. Fit scaler + produce scaled arrays + `scaler.pkl`  
   `python fit_scaler.py`

6. Build class weights + `class_weights.pkl`  
   `python class_weightbuilder.py`

---

## Why this pipeline exists

The raw data and crossing annotations are not immediately usable for machine learning. This pipeline solves that by:

- organizing the available data into a consistent index
- extracting and aligning model-ready features
- converting crossing events into dense pointwise labels
- slicing the continuous sequence data into fixed-size windows
- scaling the features for stable model training
- compensating for label imbalance through class weights

This makes the final outputs ready for downstream training, validation, and testing.

---
