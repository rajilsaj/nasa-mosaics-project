# Precision Analysis Summary

## Current Problem

**Precision is very low: 1.65% - 3.78%** across all tested thresholds (0.45 - 0.90)

### Test Results (Sliding Windows, 85,925 samples, 225:1 imbalance)

| Threshold | Precision | Recall | F1-Score | TP | FP |
|-----------|-----------|--------|----------|----|----|
| 0.45 | 1.65% | 42.63% | 3.18% | 162 | 9,642 |
| 0.60 | 2.35% | 21.84% | 4.25% | 83 | 3,445 |
| 0.75 | 2.86% | 13.42% | 4.72% | 51 | 1,731 |
| 0.90 | 3.78% | 6.58% | 4.80% | 25 | 636 |
| 0.97 | 4.55% | 0.79% | 1.35% | 3 | 63 |

**Key Issue**: Even at threshold 0.97, we only get **4.55% precision** (3 TP, 63 FP)

---

## Root Causes

### 1. **Distribution Shift**
- **Training**: Balanced data (1:1 positive:negative ratio)
- **Deployment**: Extremely imbalanced (225:1 negative:positive ratio)
- Model probabilities are calibrated for balanced data, not imbalanced deployment

### 2. **Model Calibration Issues**
- Model outputs probabilities in range [0.13, 0.99]
- These probabilities are not well-calibrated for the deployment scenario
- Even high probabilities (0.97) result in many false positives

### 3. **Extreme Class Imbalance**
- Only 380 positive samples out of 85,925 (0.44%)
- Model struggles to distinguish true positives from false positives
- Many negative samples get high probability scores

### 4. **Feature Limitations**
- Model relies on features like `second_half_slope`, `pressure_drop`, `range`
- These features may not be specific enough to vortex events
- Non-vortex pressure variations can trigger similar feature values

---

## Analysis Results

### Attempted Improvements

1. **Higher Thresholds (0.90 - 0.99)**
   - Best precision: **4.55%** at threshold 0.97
   - Still very low, with only 3 true positives

2. **Class Prior Adjustment (Bayes Rule)**
   - Adjusted probabilities too low (max 0.29)
   - No valid thresholds found after adjustment
   - **Result**: Not effective for this scenario

3. **Probability Calibration (Isotonic/Sigmoid)**
   - Failed due to NaN values in training data
   - **Result**: Could not be tested

---

## Recommendations

### Immediate Actions (Quick Wins)

#### 1. **Use Higher Thresholds**
- Current best: **Threshold 0.97** → Precision 4.55%
- Accept very low recall (0.79%) to minimize false positives
- **Trade-off**: Will miss most vortices but reduce false alarms

#### 2. **Post-Processing Filters**
Implement temporal consistency checks:
```python
# Require multiple consecutive positive predictions
if sum(consecutive_predictions) >= 3:
    trigger_vortex_detection()
```

#### 3. **Two-Stage Detection**
- **Stage 1**: Simple threshold on `pressure_drop` feature (low power)
- **Stage 2**: Random Forest (only when Stage 1 triggers)
- **Benefit**: Reduces false positives by pre-filtering

---

### Medium-Term Solutions

#### 4. **Retrain with Deployment Prior**
- Train model on data with deployment class distribution (225:1)
- Use `class_weight` parameter to match deployment priors
- **Expected**: Better calibration for imbalanced scenario

#### 5. **Cost-Sensitive Learning**
- Increase penalty for false positives during training
- Adjust `class_weight` to heavily penalize FP
- Example: `class_weight={0: 1, 1: 225}` (match deployment ratio)

#### 6. **Feature Engineering**
- Add more discriminative features:
  - Temporal patterns (pressure drop rate, recovery time)
  - Multi-window consistency
  - Environmental context (time of day, season)
- Remove or down-weight less discriminative features

---

### Long-Term Solutions

#### 7. **Alternative Models**
- **XGBoost** or **LightGBM**: Better handling of imbalanced data
- **Neural Networks**: Can learn more complex patterns
- **Ensemble Methods**: Combine multiple models for better precision

#### 8. **Active Learning**
- Deploy model with low threshold
- Collect false positives for manual review
- Retrain with corrected labels
- Iteratively improve precision

#### 9. **Hybrid Approach**
- Combine Random Forest with rule-based filters
- Use physical constraints (pressure drop magnitude, duration)
- Require multiple independent signals to agree

---

## Expected Improvements

### Conservative Estimates

| Approach | Expected Precision | Expected Recall | Implementation Time |
|----------|-------------------|-----------------|---------------------|
| Higher threshold (0.97) | 4.55% | 0.79% | ✅ Done |
| Post-processing filters | 8-12% | 2-5% | 1-2 days |
| Two-stage detection | 10-15% | 5-10% | 2-3 days |
| Retrain with deployment prior | 6-10% | 8-15% | 1 week |
| Cost-sensitive learning | 8-15% | 10-20% | 1 week |
| Feature engineering | 10-20% | 15-25% | 2-3 weeks |

### Best Case Scenario

With all improvements combined:
- **Precision**: 15-25%
- **Recall**: 20-30%
- **F1-Score**: 0.17-0.27

---

## Next Steps

1. **Immediate**: Use threshold 0.97 for deployment (4.55% precision, minimal false positives)
2. **Week 1**: Implement post-processing filters and two-stage detection
3. **Week 2-3**: Retrain model with deployment priors and cost-sensitive learning
4. **Month 1**: Feature engineering and model comparison

---

## Key Insights

1. **ROC AUC = 0.7457** shows model has reasonable discrimination ability
2. **The problem is calibration, not discrimination**
3. **Distribution shift is the main culprit** (balanced training → imbalanced deployment)
4. **Threshold optimization alone won't solve the problem** (max precision ~4.5%)
5. **Need fundamental changes**: retraining, feature engineering, or alternative models

---

## Files Generated

- `precision_improvement_analysis.py`: Comprehensive analysis script
- `results/precision_improvement_analysis_*.json`: Detailed results
- `PRECISION_ANALYSIS_SUMMARY.md`: This summary document

---

**Status**: Analysis complete  
**Best Current Precision**: 4.55% (threshold 0.97)  
**Recommendation**: Implement post-processing filters and retrain with deployment priors


