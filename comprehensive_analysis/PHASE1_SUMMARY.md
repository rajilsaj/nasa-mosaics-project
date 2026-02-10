# Phase 1: Root Cause Analysis - Summary Report

**Date**: 2025-12-30  
**Status**: ✅ COMPLETE  
**Critical Issues Found**: 5  
**Action Required**: Proceed to Phase 2

---

## Executive Summary

Phase 1 investigation has **confirmed critical data leakage** in all 4 autoencoder features. The leakage creates perfect class separation through a trivial "NaN check" rule, making the current model invalid for deployment.

---

## Section 1.1: Data Leakage Pattern Confirmed ✅

### Findings
**All 4 autoencoder features show PERFECT SEPARATION:**

| Feature | Positive Samples (NaN) | Negative Samples (NaN) | Status |
|---------|----------------------|----------------------|--------|
| `autoencoder_window_hits_mean` | 176/176 (100%) | 0/176 (0%) | 🔴 CRITICAL |
| `autoencoder_positive_hit_binary` | 176/176 (100%) | 0/176 (0%) | 🔴 CRITICAL |
| `autoencoder_hit_ratio` | 176/176 (100%) | 0/176 (0%) | 🔴 CRITICAL |
| `ae_gt_agreement` | 176/176 (100%) | 0/176 (0%) | 🔴 CRITICAL |

**Impact**: Random Forest can achieve perfect classification by simply checking `if feature.isna(): predict_positive()`

---

## Section 1.2: Feature Engineering Pipeline Investigation

### 1.2.1 Feature Engineering Logic Review
- **Location**: `feature_engineering.py`, lines ~305-353
- **Function**: `compute_autoencoder_features()`
- **Issue Found**: 
  - Returns default `0.0` values when autoencoder columns are missing
  - Computes `ae_gt_agreement` using `gt_detection_win` (ground truth) - **direct data leakage**

### 1.2.2 Window Extraction Analysis
- **Window File**: `data/windows/train_windows.csv`
- **Finding**: Autoencoder columns (`autoencoder_window_hits`, `autoencoder_positive_hit`) **ARE present** in window files
- **Sample Analysis**: 10,000 positive windows sampled, all have non-NaN autoencoder data
- **Conclusion**: The issue is NOT in window extraction - windows have the data, but it's lost during feature engineering

### 1.2.3 Source CSV Examination
- **Status**: ⚠️ Source CSV path not found at expected location
- **Expected**: `comprehensive_filtered_data_optimized.csv` in parent directory
- **Action Needed**: Verify source CSV location and check autoencoder data availability

---

## Section 1.3: Additional Issues Identified

### 1.3.1 Suspicious Features 🔴

**Critical:**
- **`ae_gt_agreement`**: Uses ground truth (`gt_detection_win`) to compute agreement - **DIRECT DATA LEAKAGE**
  - Must be removed immediately
  - Positive samples: 0 unique values (all NaN)
  - Negative samples: 2 unique values (0.0, 1.0)

**Warning:**
- **`anomaly_strength`**: Name suggests ground truth usage - needs review (likely OK, but verify)

### 1.3.2 Duplicate Features 🔴

**Identical Features:**
- **`pressure_drop`** and **`range`** are **IDENTICAL** (correlation = 1.0000)
  - **Action**: Remove one (recommend keeping `pressure_drop`, removing `range`)

**Highly Correlated Features (>0.95):**
1. `overall_slope` ↔ `mean_ratio`: 0.9983
2. `trend_consistency` ↔ `pressure_drop`: 0.9712
3. `trend_consistency` ↔ `std`: 0.9698
4. `trend_consistency` ↔ `range`: 0.9712
5. `pressure_drop` ↔ `std`: 0.9937
6. `pressure_drop` ↔ `range`: 1.0000 (identical)
7. `mean` ↔ `first_half_mean`: 0.9998
8. `mean` ↔ `second_half_mean`: 0.9998
9. `mean` ↔ `min_zscore`: 0.9995
10. `std` ↔ `range`: 0.9937
11. ... (5 more pairs)

**Action**: Consider feature selection to remove redundant features, especially:
- Remove `range` (duplicate of `pressure_drop`)
- Review `mean` vs `first_half_mean` / `second_half_mean` (very high correlation)

### 1.3.3 Systematic Missing Data Patterns 🔴

**All 4 autoencoder features show systematic missing data:**
- **Pattern**: 100% missing for positive class, 0% missing for negative class
- **Features affected**:
  1. `autoencoder_window_hits_mean`
  2. `autoencoder_positive_hit_binary`
  3. `autoencoder_hit_ratio`
  4. `ae_gt_agreement`

**Root Cause Hypothesis**: 
- Windows have autoencoder data (confirmed in Section 1.2.2)
- Feature engineering function may not be processing positive windows correctly
- OR: Positive windows are being processed differently than negative windows

---

## Key Findings Summary

| Category | Count | Severity |
|----------|-------|----------|
| **Perfect Separation (Data Leakage)** | 4 features | 🔴 CRITICAL |
| **Suspicious Features (GT Usage)** | 1 feature | 🔴 CRITICAL |
| **Duplicate Features (Identical)** | 1 pair | 🔴 CRITICAL |
| **Highly Correlated Features** | 15 pairs | ⚠️ WARNING |
| **Systematic Missing Data** | 4 features | 🔴 CRITICAL |

---

## Root Cause Hypothesis

Based on the investigation:

1. **Windows have autoencoder data** (confirmed in `train_windows.csv`)
2. **Feature engineering loses the data** for positive samples
3. **Possible causes**:
   - Positive windows are processed differently in feature engineering
   - `compute_autoencoder_features()` function has a bug when processing positive windows
   - Window extraction for positive samples doesn't preserve autoencoder columns correctly
   - Source data has autoencoder data missing for positive regions (needs verification)

**Next Step**: Investigate why positive windows lose autoencoder data during feature engineering, even though windows have the data.

---

## Recommended Actions for Phase 2

### Immediate (Critical)
1. ✅ **Remove `ae_gt_agreement` feature** (uses ground truth)
2. ✅ **Remove `range` feature** (duplicate of `pressure_drop`)
3. ✅ **Fix autoencoder feature computation** for positive windows
4. ✅ **Investigate why positive windows lose autoencoder data** during feature engineering

### High Priority
5. Review and potentially remove highly correlated features
6. Verify source CSV location and autoencoder data availability
7. Add data validation checks to prevent future leakage

### Medium Priority
8. Feature selection to reduce redundancy
9. Document feature engineering pipeline clearly

---

## Files Generated

- **Investigation Script**: `phase1_root_cause_analysis.py`
- **Findings JSON**: `results/phase1_findings_20251230_110052.json`
- **Summary Report**: `PHASE1_SUMMARY.md` (this file)

---

## Next Steps

**Proceed to Phase 2: Feature Engineering Fixes**

1. Fix `compute_autoencoder_features()` to handle positive windows correctly
2. Remove `ae_gt_agreement` from feature engineering
3. Remove `range` feature (keep `pressure_drop`)
4. Add validation checks to prevent data leakage
5. Verify source CSV and autoencoder data availability

---

**Status**: Phase 1 Complete ✅  
**Ready for**: Phase 2 Implementation



