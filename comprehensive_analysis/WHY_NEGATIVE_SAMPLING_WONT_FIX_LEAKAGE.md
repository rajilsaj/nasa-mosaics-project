# Why Negative Sampling Won't Fix the Data Leakage Problem

## The Problem

**Data Leakage**: All positive samples have NaN for autoencoder features, while all negative samples have values. This creates perfect separation.

## What Negative Sampling Does

Negative sampling:
1. ✅ Samples negative windows from safe regions
2. ✅ Calls `engineer_features_for_window()` to compute features
3. ✅ Uses the **same feature engineering function** that has the bug

## Why It Won't Fix It

### The Bug is in Feature Engineering, Not Sampling

Looking at the code flow:

```python
# negative_sampling.py line 205-210
features = engineer_features_for_window(
    window_data,
    global_mean=global_mean,
    global_std=global_std,
    include_autoencoder=include_autoencoder,  # ← Uses same function
    window_size=window_size
)
```

**The problem**: `engineer_features_for_window()` → `compute_autoencoder_features()` has a bug that causes positive windows to lose their autoencoder data.

### What Phase 1 Found

1. **Windows HAVE autoencoder data** (confirmed in `train_windows.csv`)
   - Positive windows: autoencoder columns present, non-NaN
   - Negative windows: autoencoder columns present, non-NaN

2. **Feature engineering LOSES the data** for positive samples
   - Final features: Positive samples = 100% NaN, Negative samples = 0% NaN

3. **Root cause**: Something in `compute_autoencoder_features()` or how it's called for positive windows is broken

### The Real Issue

The bug is **NOT** in:
- ❌ Window extraction (windows have the data)
- ❌ Negative sampling (it just calls feature engineering)
- ❌ Data availability (source has the data)

The bug **IS** in:
- ✅ Feature engineering function (`compute_autoencoder_features()`)
- ✅ How positive windows are processed differently than negative windows
- ✅ Possibly in how features are computed for positive vs negative samples

## What WILL Fix It

**Phase 2: Fix Feature Engineering**

1. **Fix `compute_autoencoder_features()`**:
   - Debug why positive windows lose autoencoder data
   - Ensure both positive and negative windows are processed identically
   - Handle NaN values correctly (don't create perfect separation)

2. **Remove `ae_gt_agreement`**:
   - This feature uses ground truth (`gt_detection_win`)
   - Direct data leakage - must be removed

3. **Remove duplicate features**:
   - Remove `range` (identical to `pressure_drop`)

4. **Add validation**:
   - Check that positive and negative samples have similar NaN patterns
   - Verify no perfect separation after feature engineering

## Conclusion

**Negative sampling alone will NOT solve the problem** because:

1. It uses the same broken feature engineering function
2. The bug is in feature computation, not in sampling
3. Adding more negative samples won't fix why positive samples lose their autoencoder data

**You MUST fix Phase 2 first**, then re-run negative sampling with the fixed feature engineering.

---

## Recommended Workflow

1. ✅ **Phase 1**: Root cause analysis (DONE - found the bug)
2. 🔄 **Phase 2**: Fix feature engineering (DO THIS NEXT)
3. 🔄 **Re-run negative sampling**: After Phase 2 is fixed
4. 🔄 **Phase 3**: Retrain model with fixed features

**Don't skip Phase 2!** The bug will persist even with more negative samples.



