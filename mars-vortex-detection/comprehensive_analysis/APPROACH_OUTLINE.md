# Approach Outline: Comprehensive Dataset Analysis

## Overview
This document outlines the systematic approach to building and evaluating a Random Forest classifier using `comprehensive_filtered_data_optimized.csv` with autoencoder features.

---

## Phase 1: Data Understanding & Validation

### 1.1 Dataset Exploration
- [x] Load and verify dataset integrity
- [x] Check temporal ordering (SCLK sorted)
- [ ] Analyze missing values
- [ ] Verify ground truth labels consistency
- [ ] Check for duplicate SCLK values
- [ ] Validate time gaps and continuity

### 1.2 Feature Analysis
- [x] Basic statistics for all columns
- [ ] Correlation analysis between features
- [ ] Autoencoder feature distribution analysis
- [ ] Pressure signal quality assessment
- [ ] Identify potential feature engineering opportunities

### 1.3 Class Distribution Analysis
- [x] Overall class imbalance (94.4:1)
- [ ] Temporal distribution of vortex events
- [ ] Seasonal/diurnal patterns (if applicable)
- [ ] Autoencoder vs ground truth agreement analysis

---

## Phase 2: Temporal Splitting

### 2.1 Split Strategy
- **Train**: 60% of data (early period)
- **Gap**: 0.5% buffer (~2 hours)
- **Validation**: 15% of data (middle period)
- **Gap**: 0.5% buffer
- **Test**: ~24% of data (late period)

### 2.2 Implementation Steps
1. Sort data by SCLK (ensure temporal order)
2. Calculate split indices with gaps
3. Create train/val/test splits
4. Validate temporal isolation (no overlap)
5. Save splits to CSV files

### 2.3 Validation Checks
- Verify no SCLK overlap between splits
- Check gap sizes are appropriate
- Ensure all splits maintain temporal order
- Verify class distribution in each split

---

## Phase 3: Window Extraction

### 3.1 Window Strategy
- **Window Size**: 60 samples (backward from precursor)
- **Extraction Method**: Extract 60 samples before `gt_detection_win=True` starts
- **Labeling**: Window labeled True if right boundary overlaps vortex region

### 3.2 Positive Windows
- Extract windows for each vortex event
- Ensure minimum 60 samples available before event
- Track window_id and event_sclk for traceability

### 3.3 Negative Windows
- Extract from "safe" regions (far from vortex events)
- Use buffer zones around positive events
- Balance training set (1:1 ratio) via negative sampling
- Keep validation/test sets with natural imbalance

### 3.4 Autoencoder Integration
- Include `autoencoder_window_hits` in window data
- Include `autoencoder_positive_hit` as feature
- Preserve `PRESSURE_MA_500` for baseline comparison

---

## Phase 4: Feature Engineering

### 4.1 Temporal Features (from PRESSURE)
1. **Trend Features**:
   - `overall_slope`: Linear trend across window
   - `first_half_slope`: Trend in first 30 samples
   - `second_half_slope`: Trend in last 30 samples
   - `trend_consistency`: Agreement between halves

2. **Statistical Features**:
   - `mean`, `std`, `variance`
   - `min`, `max`, `range`
   - `skewness`, `kurtosis`
   - `median`, `q25`, `q75`

3. **Pressure Dynamics**:
   - `pressure_drop`: Max - Min pressure
   - `pressure_gradient`: Rate of change
   - `deviation_from_ma`: PRESSURE - PRESSURE_MA_500
   - `relative_drop`: Drop / baseline

4. **Positional Features**:
   - `min_position`: Where minimum occurs (0-59)
   - `max_position`: Where maximum occurs
   - `drop_position`: Where drop starts

### 4.2 Autoencoder Features
1. **Direct Features**:
   - `autoencoder_window_hits`: Count (0-60)
   - `autoencoder_positive_hit`: Binary (0/1)
   - `autoencoder_hit_ratio`: hits / window_size

2. **Derived Features**:
   - `ae_agreement_with_gt`: Agreement with ground truth
   - `ae_confidence`: Normalized hit count
   - `ae_temporal_pattern`: Distribution of hits in window

### 4.3 Anomaly Features
- Z-score based anomaly indicators
- Rolling window statistics
- Deviation from moving average
- Rate of change indicators

### 4.4 Feature Selection
- Remove highly correlated features (>0.95)
- Keep features with variance > threshold
- Prioritize interpretable features
- Target: ~15-20 optimized features

---

## Phase 5: Model Training

### 5.1 Data Preparation
- Load balanced training features (1:1 ratio)
- Prepare feature matrix X and labels y
- Normalize features (z-score using training stats only)
- Split into train/validation for hyperparameter tuning

### 5.2 Random Forest Configuration
```python
RandomForestClassifier(
    n_estimators=100,          # Start with 100, tune if needed
    max_depth=12-15,           # Prevent overfitting
    min_samples_split=15,      # Require sufficient samples
    min_samples_leaf=8,        # Minimum leaf size
    max_features='sqrt',       # Feature subset size
    class_weight='balanced',   # Handle imbalance
    random_state=42,
    n_jobs=-1
)
```

### 5.3 Hyperparameter Tuning (Optional)
- Grid search or random search
- Focus on: max_depth, min_samples_split, n_estimators
- Use validation set for tuning
- Avoid overfitting to validation set

### 5.4 Training Process
1. Train on balanced training set
2. Evaluate on validation set (natural imbalance)
3. Monitor overfitting (train vs val performance)
4. Save model and metadata

---

## Phase 6: Model Evaluation

### 6.1 Evaluation Strategy
- **Training**: Balanced set (1:1) - for learning
- **Validation**: Natural imbalance - for tuning
- **Test**: Natural imbalance - for final assessment

### 6.2 Metrics to Track
- **Primary**: Precision, Recall, F1-Score
- **Secondary**: Accuracy, ROC AUC
- **Error Rates**: FPR, FNR
- **Confusion Matrix**: TP, FP, TN, FN

### 6.3 Threshold Analysis
- Evaluate at multiple thresholds: [0.45, 0.50, 0.60, 0.75, 0.90]
- Generate PR curve and ROC curve
- Identify optimal threshold for deployment
- Consider precision vs recall trade-offs

### 6.4 Sliding Window Evaluation
- Simulate real-time deployment
- Generate continuous probability predictions
- Evaluate on overlapping windows (step=10)
- Compare with fixed-window evaluation

---

## Phase 7: Feature Importance & Interpretation

### 7.1 Feature Importance Analysis
- Extract `feature_importances_` from RF
- Rank features by importance
- Visualize top 10-15 features
- Compare with physical understanding

### 7.2 Autoencoder Contribution
- Assess importance of autoencoder features
- Compare RF-only vs RF+autoencoder performance
- Determine if autoencoder adds value

### 7.3 Model Interpretation
- Analyze decision paths for sample predictions
- Identify key pressure patterns
- Validate against known vortex physics

---

## Phase 8: Comparison & Validation

### 8.1 Baseline Comparison
- Compare with previous model (temporal_splits)
- Compare with/without autoencoder features
- Assess improvement from comprehensive dataset

### 8.2 Class Prior Analysis
- Analyze class distributions across splits
- Consider deployment priors
- Evaluate EM-based probability adjustment (from Latinne paper)

### 8.3 Visualization
- Event-based plots (individual windows)
- Continuous twinx plots (pressure + probabilities)
- Confusion matrices at different thresholds
- PR/ROC curves

---

## Phase 9: Deployment Preparation

### 9.1 Model Optimization
- Finalize feature set (15 optimized features)
- Set deployment threshold
- Document model configuration
- Create inference pipeline

### 9.2 Performance Documentation
- Record all metrics (precision, recall, F1)
- Document threshold selection rationale
- Note class imbalance handling
- Include feature importance rankings

### 9.3 Reproducibility
- Save all configurations
- Document random seeds
- Version control all scripts
- Create requirements.txt

---

## Phase 10: Future Improvements

### 10.1 Potential Enhancements
- Ensemble methods (RF + Autoencoder)
- Advanced feature engineering
- Hyperparameter optimization
- Cross-validation strategies

### 10.2 Research Directions
- EM-based probability adjustment
- Active learning for rare events
- Transfer learning approaches
- Real-time adaptation strategies

---

## File Structure

```
comprehensive_analysis/
├── README.md                          # Workspace overview
├── APPROACH_OUTLINE.md                # This file
├── analyze_comprehensive_dataset.py    # Initial analysis
├── data_preparation.py                 # Temporal splitting
├── window_extraction.py                # Window extraction
├── feature_engineering.py             # Feature creation
├── train_rf_model.py                  # Model training
├── evaluate_model.py                   # Model evaluation
├── feature_importance_analysis.py     # Feature analysis
├── results/                           # Output directory
│   ├── splits/                        # Train/val/test splits
│   ├── windows/                       # Extracted windows
│   ├── features/                      # Engineered features
│   ├── models/                        # Trained models
│   └── plots/                         # Visualizations
└── data/                              # Processed data files
```

---

## Key Principles

1. **Temporal Integrity**: Never break causality, maintain chronological order
2. **No Data Leakage**: Strict temporal splits with gaps
3. **Realistic Evaluation**: Test on natural imbalance, not balanced
4. **Interpretability**: Prioritize understandable features
5. **Reproducibility**: Document everything, use random seeds
6. **Incremental Development**: Test each phase before moving forward

---

## Success Criteria

- [ ] Model achieves F1 > 0.05 on test set (improvement over baseline)
- [ ] Precision > 3% at reasonable recall (>10%)
- [ ] Autoencoder features contribute meaningfully
- [ ] Model is interpretable and explainable
- [ ] All code is documented and reproducible
- [ ] Results are clearly visualized and reported

---

## Timeline Estimate

- Phase 1-2: Data prep & splitting (1-2 days)
- Phase 3-4: Window extraction & features (2-3 days)
- Phase 5-6: Training & evaluation (2-3 days)
- Phase 7-8: Analysis & comparison (1-2 days)
- Phase 9-10: Documentation & improvements (1-2 days)

**Total**: ~1-2 weeks for complete pipeline

---

## Next Immediate Steps

1. Complete Phase 1.1 (data validation)
2. Implement Phase 2 (temporal splitting)
3. Begin Phase 3 (window extraction with autoencoder features)

