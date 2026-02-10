# Random Forest Expert Approach: Fixing Data Leakage in Autoencoder Features

## Executive Summary
**Problem**: Critical data leakage detected in autoencoder features causing perfect but invalid class separation. All positive samples have NaN values for all 4 autoencoder features, while all negative samples have actual values, creating a trivial "NaN check" rule instead of learning vortex detection patterns.

**Impact**: Model achieves perfect training performance (ROC AUC = 1.0) but is invalid for deployment. Current model must be discarded and retrained after fixing feature engineering.

---

## Phase 1: Root Cause Analysis & Validation

### 1.1 Confirm Data Leakage Pattern
- [ ] **Verify NaN distribution**: Check that positive samples have NaN for autoencoder features
- [ ] **Check negative samples**: Confirm negative samples have non-NaN values
- [ ] **Identify feature source**: Trace where autoencoder features are computed in the pipeline
- [ ] **Document leakage mechanism**: Understand why positive windows lack autoencoder data

### 1.2 Investigate Feature Engineering Pipeline
- [ ] **Review `compute_autoencoder_features()`**: Check logic in `feature_engineering.py`
- [ ] **Check window extraction**: Verify how windows are extracted in `data_preparation.py`
- [ ] **Examine source data**: Check if `comprehensive_filtered_data_optimized.csv` has autoencoder data for positive regions
- [ ] **Trace data flow**: Map from raw data → window extraction → feature engineering → training

### 1.3 Identify Additional Issues
- [ ] **Suspicious features**: Remove `ae_gt_agreement` (may contain ground truth information)
- [ ] **Duplicate features**: Check for redundant features (e.g., `pressure_drop` vs `range`)
- [ ] **Feature correlation**: Analyze correlation matrix to identify highly correlated features (>0.95)
- [ ] **Missing data patterns**: Check for other systematic missing data issues

---

## Phase 2: Feature Engineering Fixes

### 2.1 Fix Autoencoder Feature Computation
**Goal**: Ensure autoencoder features are computed correctly for BOTH positive and negative windows.

#### 2.1.1 Diagnose Missing Data Source
- [ ] Check if source CSV (`comprehensive_filtered_data_optimized.csv`) has autoencoder columns populated for positive regions
- [ ] If missing in source: Investigate why autoencoder didn't process positive regions
- [ ] If present in source: Fix window extraction to preserve autoencoder data

#### 2.1.2 Fix `compute_autoencoder_features()` Function
**Current Issue**: Function returns default values (0.0) when autoencoder columns are missing, but this creates perfect separation.

**Fix Strategy**:
```python
# Option A: If autoencoder data is truly missing for positive samples
# → Remove autoencoder features entirely (use baseline 15 features only)

# Option B: If autoencoder data exists but wasn't extracted
# → Fix window extraction to include autoencoder columns

# Option C: If autoencoder data is sparse
# → Use imputation strategy (mean/median) but flag as missing
# → Add binary indicator: 'autoencoder_data_available' (0/1)
```

**Recommended Approach**:
1. **First**: Check if autoencoder data exists in source for positive regions
2. **If yes**: Fix extraction pipeline to preserve it
3. **If no**: Remove autoencoder features and retrain with baseline 15 features
4. **If partial**: Use imputation + missing indicator, but expect lower performance

#### 2.1.3 Remove Suspicious Features
- [ ] **Remove `ae_gt_agreement`**: This feature compares autoencoder output with ground truth, creating direct leakage
- [ ] **Check for other GT-derived features**: Search for any features computed using ground truth labels
- [ ] **Validate feature independence**: Ensure all features are computed from input data only, not labels

#### 2.1.4 Handle Duplicate Features
- [ ] **Identify duplicates**: Check if `pressure_drop` and `range` are identical
- [ ] **Remove redundant features**: Keep only one (prefer `pressure_drop` as it's more interpretable)
- [ ] **Check correlation matrix**: Remove features with correlation > 0.95

---

## Phase 3: Data Quality Validation

### 3.1 Pre-Training Data Validation
**Create validation script** to check data quality before training:

- [ ] **NaN check**: Verify no systematic NaN patterns by class
- [ ] **Feature distribution**: Check that features have reasonable distributions for both classes
- [ ] **Perfect separation check**: Verify no single feature perfectly separates classes
- [ ] **Temporal integrity**: Ensure no future data leaks into past windows
- [ ] **Class balance**: Verify training set is balanced (1:1), validation/test are imbalanced

### 3.2 Feature Sanity Checks
- [ ] **Range validation**: Check that all features are within expected ranges
- [ ] **Variance check**: Ensure features have non-zero variance (remove constant features)
- [ ] **Outlier detection**: Flag extreme outliers that might indicate data errors
- [ ] **Missing data summary**: Report missing data percentage per feature per class

---

## Phase 4: Random Forest Model Retraining

### 4.1 Feature Set Selection
**Option A: Baseline Only (15 features)**
- Use original 15 proven features
- Remove all autoencoder features
- **Pros**: Clean, validated feature set
- **Cons**: Loses potential signal from autoencoder

**Option B: Fixed Autoencoder Features (if data available)**
- Fix autoencoder feature computation
- Use 15 original + 3 autoencoder features (remove `ae_gt_agreement`)
- **Pros**: Potentially better performance
- **Cons**: Requires fixing data pipeline first

**Option C: Hybrid Approach**
- Start with baseline (15 features) → establish clean baseline
- Then add fixed autoencoder features → incremental improvement
- **Pros**: Clear comparison, incremental validation
- **Cons**: More training iterations

**Recommendation**: Start with **Option A** (baseline only) to establish a clean, validated model. Then proceed to Option C if autoencoder data can be fixed.

### 4.2 Random Forest Hyperparameter Configuration
**Current Config** (from `train_with_autoencoder.py`):
```python
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': True
}
```

**Recommended Adjustments for Small Dataset** (352 samples):
```python
RF_CONFIG_FIXED = {
    'n_estimators': 100,          # Reduced: 200 may be overkill for 352 samples
    'max_depth': 10,              # Reduced: Prevent overfitting on small data
    'min_samples_split': 20,       # Increased: Require more samples per split
    'min_samples_leaf': 10,       # Increased: Larger leaves = more stable
    'max_features': 'sqrt',       # Keep: Standard for RF
    'class_weight': 'balanced',   # Keep: Handle imbalance
    'random_state': 42,           # Keep: Reproducibility
    'n_jobs': -1,                 # Keep: Parallelization
    'oob_score': True,            # Keep: Free validation
    'bootstrap': True,             # Explicit: Enable bootstrap
    'max_samples': 0.8            # NEW: Use 80% samples per tree (reduces overfitting)
}
```

**Rationale**:
- **Small dataset (352 samples)**: Need more regularization
- **Higher `min_samples_split/leaf`**: Prevent overfitting, more stable trees
- **Lower `max_depth`**: Reduce model complexity
- **`max_samples=0.8`**: Bootstrap with 80% samples adds regularization

### 4.3 Training Data Size
**Current**: 352 samples (176 positive + 176 negative)

**Recommendation**: Increase to 1,000+ samples if possible
- [ ] **Expand negative sampling**: Extract more negative windows from safe regions
- [ ] **Check data availability**: Verify if more positive windows can be extracted
- [ ] **Balance considerations**: Maintain 1:1 ratio for training, but increase total size

**Why**: More data = more stable Random Forest, better generalization, less overfitting risk.

### 4.4 Training Procedure
1. **Load fixed training data** (no data leakage)
2. **Validate data quality** (run Phase 3 checks)
3. **Train Random Forest** with adjusted hyperparameters
4. **Monitor OOB score**: Should be < 1.0 (perfect OOB = suspicious)
5. **Check feature importance**: Verify reasonable distribution (not dominated by one feature)
6. **Evaluate on training**: Should show good but not perfect performance

**Expected Training Performance** (healthy model):
- **ROC AUC**: 0.75 - 0.90 (not 1.0!)
- **OOB Score**: 0.70 - 0.85
- **Feature Importance**: Distributed across multiple features
- **Training F1**: 0.70 - 0.85 (not 1.0!)

---

## Phase 5: Model Validation & Evaluation

### 5.1 Validation Set Evaluation
**Key Principle**: Tune threshold on validation set ONLY, never on test set.

1. **Load validation data** (natural imbalance, ~88:1)
2. **Generate predictions** (probabilities)
3. **Tune threshold** on validation:
   - Test thresholds: [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
   - Find optimal threshold (F1-maximizing)
   - Save optimal threshold
4. **Evaluate with optimal threshold**:
   - Precision, Recall, F1-Score
   - Confusion matrix
   - ROC AUC, PR AUC

**Expected Validation Performance** (realistic):
- **ROC AUC**: 0.70 - 0.85
- **F1-Score** (at optimal threshold): 0.40 - 0.60
- **Precision**: 0.30 - 0.70 (depends on threshold)
- **Recall**: 0.50 - 0.80 (depends on threshold)

### 5.2 Test Set Evaluation
**Key Principle**: Use threshold from validation, do NOT retune on test.

1. **Load test data** (natural imbalance)
2. **Load optimal threshold** (from validation tuning)
3. **Evaluate with frozen threshold**:
   - Apply threshold to test probabilities
   - Compute final metrics
   - Report unbiased performance

**Expected Test Performance** (should match validation):
- **ROC AUC**: Similar to validation (±0.05)
- **F1-Score**: Similar to validation (±0.10)
- **No perfect performance**: If test F1 > 0.95, investigate for leakage

### 5.3 Probability Distribution Analysis
**Check for healthy probability distributions**:

- [ ] **Training set**: Should show separation but not perfect (max prob < 1.0)
- [ ] **Validation set**: Should show lower probabilities due to distribution shift
- [ ] **Test set**: Should match validation distribution
- [ ] **No uniform probabilities**: All positive samples should NOT have identical probabilities

**Healthy Pattern**:
- Training: Max prob ~0.80-0.95, mean ~0.50
- Validation: Max prob ~0.10-0.30, mean ~0.01-0.05
- Test: Similar to validation

---

## Phase 6: Feature Importance Analysis

### 6.1 Interpret Feature Importance
**After retraining**, analyze feature importance:

- [ ] **Top features**: Identify which features Random Forest relies on most
- [ ] **Autoencoder contribution**: If using autoencoder features, check their importance
- [ ] **Physical interpretation**: Verify top features align with known vortex physics
- [ ] **Feature stability**: Check if importance is consistent across different random seeds

**Expected Results**:
- Top features should be: `pressure_drop`, `min_zscore`, `anomaly_strength`, `overall_slope`
- Autoencoder features (if fixed) should have moderate importance (not dominating)
- No single feature should have >40% importance (indicates over-reliance)

### 6.2 Feature Selection (Optional)
**If needed**, perform feature selection:

- [ ] **Remove low-importance features**: Features with <1% importance
- [ ] **Retrain with reduced set**: Compare performance
- [ ] **Use recursive feature elimination**: If dataset is large enough

**Note**: With only 15-19 features, feature selection may not be necessary.

---

## Phase 7: Model Comparison & Documentation

### 7.1 Baseline vs. Extended Comparison
**Compare models**:
- [ ] **Baseline (15 features)**: Clean, validated model
- [ ] **Extended (15 + autoencoder)**: If autoencoder features are fixed
- [ ] **Metrics**: ROC AUC, F1-Score, Precision, Recall
- [ ] **Feature importance**: Compare which features matter most

### 7.2 Documentation
**Document the fix process**:

- [ ] **Root cause**: Document why data leakage occurred
- [ ] **Fix applied**: Describe what was changed in feature engineering
- [ ] **Model configuration**: Record hyperparameters used
- [ ] **Performance metrics**: Document training, validation, test results
- [ ] **Lessons learned**: Note what to avoid in future

### 7.3 Reproducibility
**Ensure reproducibility**:

- [ ] **Save model artifacts**: Model file, metadata, feature list
- [ ] **Save data splits**: Training, validation, test sets (with timestamps)
- [ ] **Record random seeds**: All random operations use seed=42
- [ ] **Version control**: Commit fixed scripts to Git

---

## Phase 8: Deployment Readiness Checklist

### 8.1 Pre-Deployment Validation
- [ ] **No data leakage**: Confirmed via validation scripts
- [ ] **No perfect separation**: Training ROC AUC < 1.0
- [ ] **Reasonable performance**: Validation F1 > 0.40
- [ ] **Stable probabilities**: No uniform probabilities, reasonable distributions
- [ ] **Feature importance**: Distributed, interpretable
- [ ] **Test performance**: Matches validation (no overfitting)

### 8.2 Monitoring Plan
**For production deployment**:

- [ ] **Monitor prediction distribution**: Alert if probabilities become uniform
- [ ] **Track performance metrics**: Precision, Recall, F1 over time
- [ ] **Check for distribution shift**: Compare incoming data to training distribution
- [ ] **Feature drift detection**: Monitor feature distributions for changes

---

## Implementation Priority

### High Priority (Do First)
1. ✅ **Fix autoencoder feature computation** (Phase 2.1)
2. ✅ **Remove suspicious features** (`ae_gt_agreement`) (Phase 2.1.3)
3. ✅ **Remove duplicate features** (Phase 2.1.4)
4. ✅ **Data quality validation** (Phase 3)
5. ✅ **Retrain with baseline features** (Phase 4, Option A)

### Medium Priority (After Baseline Works)
6. **Increase training data size** (Phase 4.3)
7. **Add fixed autoencoder features** (if data available) (Phase 4.1, Option C)
8. **Hyperparameter tuning** (Phase 4.2)

### Low Priority (Optimization)
9. **Feature selection** (Phase 6.2)
10. **Advanced calibration** (if needed)

---

## Key Random Forest Principles Applied

1. **No Data Leakage**: Features must be computed from input data only, never from labels
2. **Regularization for Small Data**: Higher `min_samples_split/leaf`, lower `max_depth` for 352 samples
3. **Class Weight Balancing**: Use `class_weight='balanced'` to handle imbalance
4. **OOB Validation**: Monitor OOB score - perfect OOB = suspicious
5. **Feature Importance**: Use for interpretability and validation
6. **Temporal Integrity**: Maintain chronological order, no future data in past windows
7. **Proper Splitting**: Tune threshold on validation, evaluate on test with frozen threshold

---

## Expected Timeline

- **Phase 1-2** (Root cause & fixes): 2-4 hours
- **Phase 3** (Data validation): 1-2 hours
- **Phase 4** (Retraining): 1-2 hours
- **Phase 5** (Evaluation): 1-2 hours
- **Phase 6-8** (Analysis & documentation): 2-3 hours

**Total**: ~8-13 hours for complete fix and validation

---

## Success Criteria

✅ **Model is valid** if:
- Training ROC AUC < 1.0 (not perfect)
- OOB score < 1.0
- Feature importance is distributed
- No systematic NaN patterns by class
- Validation F1 > 0.40
- Test performance matches validation
- Probability distributions are reasonable (not uniform)

❌ **Model is invalid** if:
- Training ROC AUC = 1.0 (perfect separation)
- OOB score = 1.0
- Single feature dominates importance (>50%)
- Systematic NaN patterns by class
- All positive samples have identical probabilities
- Test performance >> validation (overfitting)

---

## Notes

- **Current model is invalid** and must be discarded
- **Start fresh** with clean feature engineering
- **Prioritize correctness over performance** - a valid 70% F1 is better than an invalid 100% F1
- **Document everything** for reproducibility and learning



