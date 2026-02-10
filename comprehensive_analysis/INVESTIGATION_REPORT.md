# Investigation Report: Suspicious Training Results

**Date:** 2025-12-19  
**Investigation:** Extended Model Training Results Analysis

---

## Executive Summary

The investigation revealed **CRITICAL DATA LEAKAGE** in the autoencoder features, which explains all suspicious results:
- Perfect training performance (ROC AUC = 1.0)
- All positive samples getting identical probability (99.74%)
- Severe overfitting

---

## Critical Finding: Data Leakage

### The Problem

**All 4 autoencoder features have perfect separation:**

| Feature | Positive Samples | Negative Samples |
|---------|----------------|------------------|
| `autoencoder_window_hits_mean` | **100% NaN** | **100% have values** |
| `autoencoder_positive_hit_binary` | **100% NaN** | **100% have values** |
| `autoencoder_hit_ratio` | **100% NaN** | **100% have values** |
| `ae_gt_agreement` | **100% NaN** | **100% have values** |

### Why This Causes Perfect Separation

The Random Forest model can learn a trivial rule:
- **If autoencoder feature is NaN → Predict POSITIVE (vortex)**
- **If autoencoder feature has value → Predict NEGATIVE (non-vortex)**

This creates **perfect class separation** without learning any actual patterns!

---

## Additional Issues Found

### 1. Suspicious Feature: `ae_gt_agreement`
- **Name suggests label leakage**: "gt" likely means "ground truth"
- **Value distribution**: 172 negative samples = 1.0, 4 negative samples = 0.0
- **All positive samples**: NaN
- **Risk**: This feature may directly encode ground truth information

### 2. Duplicate Features
- `pressure_drop` and `range` are identical
- Redundant information

### 3. Small Training Set
- Only 352 samples (176 positive, 176 negative)
- With 19 features, this is a small sample size
- High risk of overfitting even without data leakage

### 4. Overfitting Indicators
- **Training ROC AUC**: 1.0000 (perfect)
- **Validation ROC AUC**: 0.7885 (good, but much lower)
- **Performance drop**: 0.2115 (21 percentage points)
- **All positive samples fall into same leaves** in multiple trees

---

## Root Cause Analysis

### Why Do Positive Samples Have NaN?

The autoencoder features are computed from window data that includes `autoencoder_window_hits` and `autoencoder_positive_hit` columns. 

**Hypothesis:**
1. Positive windows may not have autoencoder data available
2. Or the feature engineering process fails for positive windows
3. Or positive windows are extracted differently and don't include autoencoder columns

**Evidence:**
- Feature engineering code returns `0.0` when autoencoder columns are missing
- But training data shows `NaN` for all positive samples
- This suggests the feature engineering failed or wasn't applied to positive windows

---

## Impact Assessment

### Current Model Performance is Invalid

1. **Training metrics are meaningless** - Model learned a trivial rule (NaN check)
2. **Validation metrics are misleading** - Model still works because NaN pattern persists
3. **Model cannot generalize** - It relies on data leakage, not actual patterns
4. **Deployment will fail** - If autoencoder features are computed correctly in deployment, the NaN pattern won't exist

---

## Recommendations

### Immediate Actions

1. **Remove autoencoder features from training** (or fix the NaN issue)
   - Retrain model with only 15 original features
   - Verify no NaN values in any features

2. **Fix autoencoder feature engineering**
   - Investigate why positive windows don't have autoencoder data
   - Ensure all windows (positive and negative) have autoencoder features computed
   - Replace NaN with appropriate default values (e.g., 0.0) if autoencoder data unavailable

3. **Remove suspicious features**
   - Remove `ae_gt_agreement` (likely contains ground truth information)
   - Remove duplicate feature (`range` or `pressure_drop`)

4. **Increase training data**
   - Current 352 samples is too small
   - Aim for at least 1,000+ samples to reduce overfitting risk

### Long-term Actions

1. **Implement data validation pipeline**
   - Check for NaN values in features
   - Check for perfect feature-label correlations
   - Check for suspicious feature names

2. **Add feature importance monitoring**
   - Alert if single feature dominates (>50% importance)
   - Alert if features have perfect separation

3. **Improve cross-validation**
   - Use stratified k-fold cross-validation
   - Monitor train/validation gap

---

## Conclusion

The suspicious results are **100% explained by data leakage** in autoencoder features. The model learned a trivial rule (NaN check) rather than actual vortex detection patterns. 

**The current model is not valid for deployment** and must be retrained after fixing the data leakage issues.

---

## Files Generated

- `investigate_suspicious_results.py` - Investigation script
- `deep_dive_autoencoder_features.py` - Autoencoder feature analysis
- `results/investigation_results_20251219_114135.json` - Detailed results




