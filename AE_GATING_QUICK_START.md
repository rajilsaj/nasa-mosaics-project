# Autoencoder Gating - Quick Start Guide

## Overview
This experiment implements autoencoder gating for the ML dataset model to improve precision and F1-score.

**Baseline Performance:**
- Precision: 3.78%
- Recall: 6.58%
- F1-Score: 4.80%
- ROC AUC: 0.7457

**Target Performance:**
- Precision: 5-8% (improvement)
- Recall: 7-10% (maintain or improve)
- F1-Score: 6-8% (improvement)

---

## Prerequisites

### Required Files:
1. `train_windows.csv` - Training windows (60 samples per window)
2. `train_features.csv` - Training features (already engineered)
3. `test_features.csv` - Test features (for evaluation)

### If `train_windows.csv` doesn't exist:
Run the data preparation script first:
```bash
python "core pipeline scripts/data_preparation.py"
```

This will create `train_windows.csv`, `val_windows.csv`, and `test_windows.csv`.

---

## Running the Experiment

### Step 1: Run the AE Gating Script
```bash
python ml_dataset_ae_gating.py
```

### What the Script Does:
1. **Loads training windows** - For autoencoder training
2. **Trains autoencoder** - Simple MLP autoencoder on pressure windows
3. **Scores windows** - Computes reconstruction error for each window
4. **Filters training data** - Keeps top 50% by AE score (configurable)
5. **Retrains RF model** - On filtered training data
6. **Evaluates on test set** - Compares to baseline
7. **Saves results** - Models and metrics saved to `models/` and `results/`

---

## Configuration Options

Edit `ml_dataset_ae_gating.py` to change:

### Filtering Strategy:
```python
FILTER_METHOD = "top_percentile"  # Options: "top_percentile", "threshold"
FILTER_PERCENTILE = 50  # Keep top 50% by AE score
```

### Autoencoder Architecture:
```python
AE_HIDDEN_LAYERS = (32, 16, 32)  # Encoder-decoder
AE_MAX_ITER = 500
```

---

## Expected Output

### Console Output:
```
======================================================================
ML DATASET MODEL WITH AUTOENCODER GATING
======================================================================
Date: 2025-01-XX XX:XX:XX
Baseline: Precision 0.0378, F1 0.0480
======================================================================

[Step 1: Loading training windows...]
[Step 2: Training autoencoder...]
[Step 3: Scoring windows...]
[Step 4: Filtering training data...]
[Step 5: Training RF model...]
[Step 6: Evaluating on test set...]

======================================================================
COMPARISON TO BASELINE
======================================================================

Metric          Baseline     New Model    Change       Status
----------------------------------------------------------------------
Precision       0.0378       0.XXXX       +X.XXXX      ✅ IMPROVED
Recall          0.0658       0.XXXX       +X.XXXX      ✅ IMPROVED
F1_score        0.0480       0.XXXX       +X.XXXX      ✅ IMPROVED
Roc_auc         0.7457       0.XXXX       +X.XXXX      ✅ IMPROVED

======================================================================
✅ OVERALL: IMPROVEMENT DETECTED
======================================================================
```

### Saved Files:
- `models/ae_ml_dataset_TIMESTAMP.pkl` - Trained autoencoder
- `models/rf_ae_gated_ml_TIMESTAMP.pkl` - Retrained RF model
- `results/ae_window_scores_TIMESTAMP.json` - Window scores
- `results/ae_gating_results_TIMESTAMP.json` - Full results

---

## Interpreting Results

### Success Criteria:
- ✅ **F1-Score > 5.0%** - Improvement over baseline (4.80%)
- ✅ **Precision > 4.0%** - Improvement over baseline (3.78%)
- ✅ **Recall ≥ 6.0%** - Maintain or improve detection rate
- ✅ **ROC AUC ≥ 0.74** - Maintain ranking ability

### If Results Are Better:
- ✅ **Adopt the model** - Use for deployment
- ✅ **Document improvements** - Update conference paper
- ✅ **Consider tuning** - Try different filter percentiles (40%, 60%)

### If Results Are Worse:
- ❌ **Revert to baseline** - Keep original model
- 🔍 **Analyze why** - Check filtering logic, AE performance
- 🔄 **Try different settings** - Adjust filter percentile or threshold

---

## Troubleshooting

### Error: "train_windows.csv not found"
**Solution:** Run data preparation script first:
```bash
python "core pipeline scripts/data_preparation.py"
```

### Error: "No valid pressure windows found"
**Solution:** Check that `train_windows.csv` has `PRESSURE` column and `window_id` column.

### Error: "No samples have AE scores"
**Solution:** Check that `window_id` in `train_features.csv` matches `window_id` in `train_windows.csv`.

### Poor Results:
1. **Try different filter percentile** - Change `FILTER_PERCENTILE` to 40 or 60
2. **Check AE performance** - Verify autoencoder reconstruction error is reasonable
3. **Verify data quality** - Ensure training windows are valid

---

## Next Steps After Experiment

### If Successful:
1. ✅ **Document results** - Add to conference paper
2. ✅ **Compare with comprehensive model** - See which performs better
3. ✅ **Deploy best model** - Use for actual vortex detection

### If Unsuccessful:
1. 🔄 **Try comprehensive model + AE gating** - May work better
2. 🔄 **Try other improvements** - Feature removal, sliding windows
3. 🔄 **Analyze failure** - Understand why AE gating didn't help

---

## Time Estimate

- **Autoencoder training**: ~5-10 minutes
- **Window scoring**: ~1-2 minutes
- **RF training**: ~30 seconds
- **Evaluation**: ~1 minute
- **Total**: ~10-15 minutes

---

## Questions?

Check the script comments or review the baseline comparison in the results file.

**Good luck with the experiment!** 🚀
