# Why Different Thresholds Are Needed: Probability Distribution Analysis

## Overview

This document explains why the original model and comprehensive models require different decision thresholds, and why comparing them at their respective optimal thresholds is the fairest approach.

## Key Findings

### Probability Distribution Statistics

**Comprehensive Baseline Model:**
- Min: 0.0000
- Max: 0.1240 (12.4%)
- Mean: 0.0036 (0.36%)
- Median: ~0.002

**Comprehensive Autoencoder Model:**
- Min: 0.0000
- Max: 0.0288 (2.88%)
- Mean: 0.0026 (0.26%)
- Median: ~0.002

**Original Model (from ml_ready_vortex_data.csv):**
- Probabilities range up to 0.9+ (90%+)
- Can effectively use thresholds 0.45-0.90

## Why Different Thresholds?

### 1. **Probability Scale Mismatch**

The comprehensive models produce **much lower probabilities** than the original model:

- **Original Model**: Probabilities can reach 0.9+ (90%+)
- **Comprehensive Baseline**: Maximum probability = 12.4%
- **Comprehensive Autoencoder**: Maximum probability = 2.88%

### 2. **At Original Thresholds (0.45-0.90)**

When we apply the original model's thresholds to comprehensive models:

| Threshold | Baseline Predictions | Autoencoder Predictions |
|-----------|---------------------|------------------------|
| 0.45 | 0 (all negative) | 0 (all negative) |
| 0.60 | 0 (all negative) | 0 (all negative) |
| 0.75 | 0 (all negative) | 0 (all negative) |
| 0.90 | 0 (all negative) | 0 (all negative) |

**Result**: All predictions are negative because no probabilities exceed these thresholds.

### 3. **At Optimal Thresholds**

Each model performs best at different thresholds:

| Model | Optimal Threshold | F1-Score | Precision | Recall |
|-------|-------------------|----------|-----------|--------|
| Original | 0.90 | 4.80% | 3.78% | 6.58% |
| Baseline | 0.02 | 2.22% | 1.28% | 8.37% |
| Autoencoder | 0.01 | 4.89% | 2.92% | 15.16% |

## Why Are Comprehensive Models More Conservative?

### Possible Reasons:

1. **Smaller Training Dataset**
   - Comprehensive: ~1.0M samples
   - Original: ~2.5M samples
   - Less data → more uncertainty → lower probabilities

2. **Different Data Distribution**
   - Comprehensive dataset is more filtered/optimized
   - May have different characteristics
   - Model learns different patterns

3. **Autoencoder Features**
   - Autoencoder model is especially conservative (max 2.88%)
   - Autoencoder features may provide more nuanced signals
   - Lower probabilities but better discrimination (ROC AUC = 79.33%)

4. **Class Imbalance**
   - Comprehensive sliding windows: 90:1 ratio
   - Original sliding windows: 225:1 ratio
   - Different imbalance → different probability calibration

## Fair Comparison Strategy

### Option 1: Same Thresholds (Unfair)
- Apply 0.45-0.90 to all models
- **Result**: Original wins (comprehensive models predict all negatives)
- **Problem**: Doesn't reflect true model capabilities

### Option 2: Optimal Thresholds (Fair) ✅
- Compare each model at its best operating point
- **Result**: Autoencoder wins (better F1, Recall, ROC AUC)
- **Advantage**: Reflects true model performance

## Visualizations Created

Two visualization files have been generated:

1. **`probability_distributions_*.png`**
   - Comprehensive probability histograms
   - Cumulative distributions
   - Box plots
   - Statistics tables
   - Threshold comparison plots

2. **`threshold_explanation_*.png`**
   - Focused explanation plot
   - Probability range comparison
   - Predictions at different thresholds
   - Explanation text

## Conclusion

**Different thresholds are needed because:**
1. Models produce different probability distributions
2. Comprehensive models are more conservative (lower probabilities)
3. Each model's optimal threshold reflects its calibration

**Fair comparison approach:**
- Compare at **optimal thresholds** (each model's best)
- Autoencoder model wins: Better F1 (4.89% vs 4.80%), Recall (15.16% vs 6.58%), and ROC AUC (79.33% vs 74.57%)

**Key Insight:**
The autoencoder model's lower probabilities don't indicate poor performance—they indicate a different calibration. When evaluated at the appropriate threshold (0.01), it outperforms the original model.






