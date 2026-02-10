# Expert Guide: Building a Production-Quality Random Forest Model
## Step-by-Step Process from Data to Deployment

---

## 🎯 **Expert's Approach: Build Incrementally, Validate Continuously**

As a seasoned RF expert, I recommend this **incremental approach**:
1. **Establish baseline** (proven features only)
2. **Add autoencoder** (measure contribution)
3. **Optimize** (hyperparameters, thresholds)
4. **Refine** (error analysis, feature selection)

---

## 📋 **Phase 1: Data Preparation & Validation (CRITICAL)**

### **Step 1.1: Create Temporal Splits**
**Why**: Proper temporal splitting prevents data leakage and ensures realistic evaluation.

```bash
cd comprehensive_analysis
python data_preparation.py --extract_windows --window_size 60
```

**What to Check**:
- ✅ Splits created in `data/splits/`
- ✅ Windows extracted in `data/windows/`
- ✅ Temporal gaps present (no overlap)
- ✅ Autoencoder features preserved in windows

**Validation Script Needed**: `validate_data_preparation.py`

---

### **Step 1.2: Data Quality Checks**
**Why**: Garbage in = garbage out. Validate before feature engineering.

**Checks**:
- Temporal continuity (no large gaps)
- SCLK monotonicity (sorted)
- Missing values (handle if any)
- Class distribution in each split
- Autoencoder feature distribution

**Script Needed**: `validate_data_quality.py`

---

## 🔬 **Phase 2: Feature Engineering (FOUNDATION)**

### **Step 2.1: Baseline Features (Your 15)**
**Why**: Start with what works. Don't break what's working.

**Features to Engineer**:
1. Trend Features (4): overall_slope, first_half_slope, second_half_slope, trend_consistency
2. Pressure Drop (3): pressure_drop, min_position, relative_drop
3. Statistics (3): mean, std, range
4. Temporal (3): mean_ratio, variance_ratio, anomaly_strength
5. Anomaly (2): Additional anomaly metrics

**Script**: `feature_engineering.py` (baseline version)

---

### **Step 2.2: Add Autoencoder Features (INCREMENTAL)**
**Why**: Add new features incrementally to measure contribution.

**New Features**:
- `autoencoder_window_hits_mean`: Mean hits in window
- `autoencoder_positive_hit_binary`: Any positive hit? (0/1)
- `autoencoder_hit_ratio`: Total hits / window_size
- `ae_gt_agreement`: Does AE agree with ground truth? (for analysis)

**Script**: `feature_engineering.py` (with autoencoder)

**Strategy**: 
- Create baseline features first
- Then add autoencoder features
- Compare performance to measure contribution

---

### **Step 2.3: Feature Selection**
**Why**: Remove redundant/correlated features. RF works best with 15-20 features.

**Process**:
1. Calculate correlation matrix
2. Remove highly correlated features (>0.95)
3. Check feature importance after training
4. Remove low-importance features (<0.01)

**Script**: `feature_selection.py`

---

## 🌳 **Phase 3: Model Training (CORE)**

### **Step 3.1: Baseline Model (No Autoencoder)**
**Why**: Establish baseline to measure improvements.

**Configuration**:
```python
RandomForestClassifier(
    n_estimators=200,        # More trees = better
    max_depth=15,            # Prevent overfitting
    min_samples_split=10,    # Require sufficient samples
    min_samples_leaf=5,      # Minimum leaf size
    max_features='sqrt',     # Standard
    class_weight='balanced', # Handle imbalance
    random_state=42,
    n_jobs=-1,
    oob_score=True           # Out-of-bag validation
)
```

**Training Data**: Balanced 1:1 (from negative sampling)
**Validation**: Natural imbalance (from comprehensive dataset)

**Script**: `train_baseline_model.py`

**Metrics to Track**:
- Training F1, Precision, Recall
- Validation F1, Precision, Recall
- OOB score
- Feature importance

---

### **Step 3.2: Model with Autoencoder Features**
**Why**: Measure autoencoder contribution.

**Same config, but**:
- Add autoencoder features to feature set
- Compare with baseline

**Script**: `train_with_autoencoder.py`

**Key Question**: Does autoencoder improve performance?
- If YES: Keep it
- If NO: Analyze why (maybe needs different features)

---

### **Step 3.3: Hyperparameter Tuning (OPTIONAL)**
**Why**: Fine-tune for optimal performance.

**Parameters to Tune**:
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 12, 15, 18]
- `min_samples_split`: [5, 10, 15, 20]
- `min_samples_leaf`: [3, 5, 8]

**Method**: RandomizedSearchCV or Optuna

**Script**: `hyperparameter_tuning.py`

**Note**: Only tune if baseline is good. Don't overfit to validation.

---

## 📊 **Phase 4: Class Prior Integration**

### **Step 4.1: Calculate Priors**
**Why**: Know deployment distribution for probability adjustment.

```python
# From comprehensive dataset:
deployment_prior_pos = 0.0105  # 1.05%
deployment_prior_neg = 0.9895  # 98.95%
```

**Script**: `calculate_class_priors.py`

---

### **Step 4.2: Probability Adjustment**
**Why**: Adjust model outputs to match deployment priors.

**Method**: Bayes adjustment (from Latinne paper)

```python
# Adjust probabilities from training prior (0.5) to deployment (0.0105)
y_proba_adjusted = adjust_probabilities_bayes(
    y_proba_original,
    training_prior=0.5,
    deployment_prior=0.0105
)
```

**Script**: `class_prior_adjustment.py`

**Apply to**: Validation and test sets only

---

### **Step 4.3: Threshold Optimization**
**Why**: Find optimal threshold for deployment.

**Process**:
1. Get adjusted probabilities on validation set
2. Test multiple thresholds: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
3. Calculate F1, Precision, Recall at each threshold
4. Choose threshold that maximizes F1 (or based on business needs)

**Script**: `optimize_threshold.py`

---

## 🎯 **Phase 5: Evaluation & Analysis**

### **Step 5.1: Comprehensive Evaluation**
**Why**: Understand model performance across all metrics.

**Metrics to Report**:
- Precision, Recall, F1-Score
- Accuracy, ROC AUC
- FPR, FNR
- Confusion Matrix
- PR Curve, ROC Curve

**At Multiple Thresholds**: [0.45, 0.60, 0.75, 0.90]

**Script**: `comprehensive_evaluation.py`

---

### **Step 5.2: Feature Importance Analysis**
**Why**: Understand what drives predictions.

**Analysis**:
- Global feature importance (default)
- Permutation importance (more robust)
- Top 10 features visualization
- Autoencoder feature contribution

**Script**: `feature_importance_analysis.py`

---

### **Step 5.3: Error Analysis**
**Why**: Learn from mistakes to improve.

**Analysis**:
- False Positives: Where does model predict vortex but none exists?
- False Negatives: Where are vortices missed?
- Autoencoder disagreement: Cases where AE and RF disagree
- Temporal patterns: Are errors clustered in time?

**Script**: `error_analysis.py`

---

## 🔄 **Phase 6: Iterative Refinement**

### **Step 6.1: Feature Engineering Based on Errors**
**Why**: Engineer features to address failure modes.

**Process**:
1. Analyze error cases
2. Identify patterns
3. Engineer new features
4. Retrain and evaluate

**Example**: If FPs occur during pressure spikes, add spike detection feature.

---

### **Step 6.2: Model Comparison**
**Why**: Choose best model configuration.

**Compare**:
- Baseline (15 features, no autoencoder)
- With autoencoder (18 features)
- With autoencoder + prior adjustment
- Best hyperparameters

**Script**: `compare_models.py`

---

## 📋 **Recommended Execution Order**

### **Week 1: Foundation**
1. ✅ Data preparation & validation
2. ✅ Feature engineering (baseline 15)
3. ✅ Train baseline model
4. ✅ Establish baseline metrics

### **Week 2: Enhancement**
5. ✅ Add autoencoder features
6. ✅ Train with autoencoder
7. ✅ Measure improvement
8. ✅ Class prior integration

### **Week 3: Optimization**
9. ✅ Hyperparameter tuning (if needed)
10. ✅ Threshold optimization
11. ✅ Comprehensive evaluation
12. ✅ Feature importance analysis

### **Week 4: Refinement**
13. ✅ Error analysis
14. ✅ Feature refinement
15. ✅ Final model selection
16. ✅ Documentation

---

## 🎓 **Expert Tips**

### **Do's** ✅
1. **Start simple**: Baseline first, then add complexity
2. **Validate continuously**: Check at each step
3. **Measure contributions**: Know what helps
4. **Monitor overfitting**: Use OOB score, train vs val
5. **Document everything**: Reproducibility matters

### **Don'ts** ❌
1. **Don't skip validation**: Data quality is critical
2. **Don't add features blindly**: Measure contribution
3. **Don't overfit to validation**: Keep test set pristine
4. **Don't ignore class imbalance**: It's extreme (94:1)
5. **Don't use default threshold**: Optimize for deployment

---

## 🚀 **Quick Start: What to Do Right Now**

### **Immediate Actions (Today)**

1. **Run Data Preparation**:
   ```bash
   cd comprehensive_analysis
   python data_preparation.py --extract_windows
   ```

2. **Validate Data Quality**:
   ```bash
   python validate_data_quality.py
   ```

3. **Engineer Baseline Features**:
   ```bash
   python feature_engineering.py --split train
   python feature_engineering.py --split val
   python feature_engineering.py --split test
   ```

4. **Train Baseline Model**:
   ```bash
   python train_baseline_model.py
   ```

5. **Evaluate Baseline**:
   ```bash
   python evaluate_baseline.py
   ```

---

## 📊 **Success Criteria**

### **Baseline Model (15 features)**
- F1-Score: > 0.04 (match or beat original)
- Precision: > 3%
- Recall: > 5%

### **With Autoencoder (18 features)**
- F1-Score: > 0.05 (improvement from baseline)
- Precision: > 4%
- Recall: > 8%

### **With Prior Adjustment**
- F1-Score: > 0.06 (best performance)
- Precision: > 5%
- Recall: > 10%

---

## 🎯 **My Expert Recommendation**

**Start Here**:
1. ✅ Data preparation (get splits and windows)
2. ✅ Baseline features (your proven 15)
3. ✅ Baseline model (establish performance)
4. ✅ Add autoencoder (measure improvement)
5. ✅ Class prior adjustment (optimize for deployment)

**Then Iterate**:
- Analyze errors
- Refine features
- Optimize hyperparameters
- Finalize model

**Key Principle**: Build incrementally, validate at each step, measure contributions.

---

**Ready to start? Let's begin with data preparation!**

