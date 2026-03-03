# Sliding Window Evaluation Summary

**Date**: 2025-11-20  
**Evaluation Type**: Sliding Windows (Deployment Simulation)  
**Step Size**: 10 samples

---

## 📊 **Key Findings**

### **1. Model Ranking Ability (ROC AUC)**

**Validation Set:**
- **Baseline Model**: ROC AUC = 0.5015 (essentially random)
- **Autoencoder Model**: ROC AUC = 0.7885 (**+0.29 improvement**)

**Test Set:**
- **Baseline Model**: ROC AUC = 0.5050 (essentially random)
- **Autoencoder Model**: ROC AUC = 0.7933 (**+0.29 improvement**)

**✅ Key Insight**: The autoencoder model has **excellent ranking ability** (0.79 ROC AUC), meaning it can distinguish positive from negative examples well. However, it's being too conservative in its predictions.

---

### **2. Deployment Performance (Realistic Imbalance)**

**Validation Set (25,191 windows, 88.6:1 imbalance):**

| Model | Threshold | Precision | Recall | F1-Score | ROC AUC |
|-------|-----------|-----------|--------|----------|---------|
| Baseline | 0.10 | 0.0000 | 0.0000 | 0.0000 | 0.5015 |
| Autoencoder | 0.10 | 0.0000 | 0.0000 | 0.0000 | **0.7885** |

**Test Set (40,354 windows, 90.3:1 imbalance):**

| Model | Threshold | Precision | Recall | F1-Score | ROC AUC |
|-------|-----------|-----------|--------|----------|---------|
| Baseline | 0.10 | 0.0556 | 0.0023 | 0.0043 | 0.5050 |
| Autoencoder | 0.10 | 0.0000 | 0.0000 | 0.0000 | **0.7933** |

---

## 🔍 **Analysis**

### **Why Low Precision/Recall?**

1. **Extreme Class Imbalance**: 88-90:1 negative-to-positive ratio
   - Training: 1:1 balanced
   - Deployment: 88-90:1 imbalanced
   - **Distribution shift** causes poor calibration

2. **Threshold Issue**:
   - Default threshold (0.5) is too high for imbalanced deployment
   - Even optimized threshold (0.1) predicts almost all negatives
   - Need even lower threshold (e.g., 0.01-0.05) for deployment

3. **Model Conservatism**:
   - Autoencoder model has excellent ROC AUC (0.79) but predicts all negatives
   - Suggests probabilities are well-ranked but need aggressive threshold lowering

---

### **Why Autoencoder Model is Better?**

1. **ROC AUC**: 0.79 vs 0.50 (baseline)
   - Excellent ranking ability
   - Can distinguish positive from negative examples

2. **Feature Importance**: Autoencoder features are top contributors
   - `autoencoder_hit_ratio`: #1 most important feature
   - 4 of top 8 features are autoencoder-related

3. **Better Model**: Autoencoder model learned better patterns

---

## 🎯 **Recommendations**

### **1. Threshold Calibration**

**Problem**: Models predict all negatives at threshold 0.1

**Solution**: 
- Test much lower thresholds (0.01, 0.02, 0.05)
- Target specific precision requirement (e.g., 90% precision)
- Accept lower recall to minimize false positives

**Expected**: Lower threshold will unlock autoencoder model's excellent ranking ability

---

### **2. Hybrid Deployment Strategy** (Recommended)

**Stage 1: Simple Threshold Detector** (Low Power, Always On)
```
If pressure_drop > threshold:
    → Trigger Stage 2
```

**Stage 2: Random Forest Classifier** (Higher Power, On-Demand)
```
If RF probability > 0.01:
    → Trigger Stage 3
```

**Stage 3: High-Rate Data Collection** (Highest Power, Rare Events)
```
Only when vortex detected
→ Capture full vortex event
```

**Benefits**:
- Low power consumption (most of the time)
- High precision (two-stage filtering)
- Catches rare events (vortices)

---

### **3. Probability Calibration**

**Problem**: Model probabilities not calibrated for deployment priors

**Solution**: 
- Apply Platt scaling or Isotonic regression
- Calibrate on validation sliding windows
- Adjust for deployment class priors (88-90:1)

---

## 📈 **Comparison with Fixed Windows**

### **Fixed Windows (Training-like Scenario)**
- **Validation**: F1=0.73, Precision=0.67, Recall=0.79
- **Test**: F1=0.80, Precision=0.71, Recall=0.91

### **Sliding Windows (Deployment Scenario)**
- **Validation**: F1=0.00, Precision=0.00, Recall=0.00 (at threshold 0.1)
- **Test**: F1=0.00-0.004, Precision=0.00-0.06, Recall=0.00-0.002 (at threshold 0.1)

**Key Insight**: 
- Models work well on **fixed windows** (aligned with training)
- Models struggle on **sliding windows** (realistic deployment) due to extreme imbalance

**However**: 
- ROC AUC shows models have excellent ranking ability
- Need lower thresholds for deployment

---

## 🔧 **Next Steps**

1. **Detailed Threshold Analysis**
   - Test thresholds: 0.01, 0.02, 0.05, 0.10, 0.20
   - Generate precision-recall curves
   - Find optimal threshold for deployment requirements

2. **Probability Calibration**
   - Calibrate probabilities for deployment priors
   - Apply Platt scaling or Isotonic regression

3. **Hybrid Strategy Design**
   - Design two-stage detection system
   - Optimize for power consumption vs detection rate

---

## 📁 **Files Created**

### **Sliding Windows:**
- `data/sliding_windows/val_sliding_windows_step10.csv` (25,191 windows)
- `data/sliding_windows/test_sliding_windows_step10.csv` (40,354 windows)

### **Sliding Features:**
- `data/features/val_sliding_features_step10.csv` (25,191 feature vectors)
- `data/features/test_sliding_features_step10.csv` (40,354 feature vectors)

### **Evaluation Results:**
- `results/val_sliding_evaluation_20251120_191016.json`
- `results/test_sliding_evaluation_20251120_191033.json`

---

**Status**: ✅ Sliding window evaluation complete  
**Key Finding**: Autoencoder model has excellent ROC AUC (0.79) but needs lower threshold for deployment  
**Next**: Detailed threshold analysis and probability calibration

