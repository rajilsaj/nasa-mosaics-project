# Dataset Comparison: Original vs Comprehensive Results

**Date**: 2025-11-20  
**Comparison**: `ml_ready_vortex_data.csv` vs `comprehensive_filtered_data_optimized.csv`

---

## 📊 **Test Set Results Comparison**

### **1. Fixed Windows Evaluation (Training-like Scenario)**

| Dataset | Model | F1-Score | Precision | Recall | ROC AUC | Threshold |
|---------|-------|----------|-----------|--------|---------|-----------|
| **Original (ml_ready)** | RF (15 features) | **0.8000** | **0.7143** | **0.9091** | **0.9849** | 0.5 |
| **Comprehensive** | Baseline (15 features) | 0.0000 | 0.0000 | 0.0000 | 0.9963 | 0.5 |
| **Comprehensive** | Autoencoder (19 features) | 0.0000 | 0.0000 | 0.0000 | **1.0000** | 0.5 |

**Analysis**:
- ✅ **Original dataset had better fixed-window performance** (F1=0.80)
- ⚠️ **Comprehensive dataset shows threshold issues** (all predictions negative at threshold 0.5)
- ✅ **ROC AUC is excellent** (0.9963-1.0) - models can rank well, but need threshold tuning

**Note**: Comprehensive results at threshold 0.5 are misleading - need threshold optimization.

---

### **2. Sliding Windows Evaluation (Deployment Scenario)**

#### **Original Dataset (ml_ready_vortex_data.csv)**

**Test Set (85,925 sliding windows, 380 positive, 85,545 negative):**

| Threshold | Precision | Recall | F1-Score | ROC AUC |
|-----------|-----------|--------|----------|---------|
| 0.45 | 1.65% | 42.63% | 3.18% | 0.7457 |
| 0.60 | 2.35% | 21.84% | 4.25% | 0.7457 |
| 0.75 | 2.86% | 13.42% | 4.72% | 0.7457 |
| **0.90** | **3.78%** | **6.58%** | **4.80%** | 0.7457 |

**Best Performance (threshold=0.90)**:
- Precision: **3.78%**
- Recall: **6.58%**
- F1-Score: **4.80%**
- ROC AUC: **0.7457**

---

#### **Comprehensive Dataset (comprehensive_filtered_data_optimized.csv)**

**Test Set (40,354 sliding windows, 442 positive, 39,912 negative):**

**Baseline Model (15 features, optimal threshold=0.02):**
- Precision: **1.28%**
- Recall: **8.37%**
- F1-Score: **2.22%**
- ROC AUC: **0.5050**

**Autoencoder Model (19 features, optimal threshold=0.01):**
- Precision: **2.92%**
- Recall: **15.16%**
- F1-Score: **4.89%**
- ROC AUC: **0.7933**

---

## 🔍 **Direct Comparison**

### **Sliding Windows (Deployment Scenario)**

| Metric | Original (ml_ready) | Comprehensive Baseline | Comprehensive Autoencoder | Winner |
|--------|---------------------|------------------------|---------------------------|--------|
| **F1-Score** | **4.80%** | 2.22% | 4.89% | **Autoencoder** ✅ |
| **Precision** | **3.78%** | 1.28% | 2.92% | **Original** ✅ |
| **Recall** | 6.58% | 8.37% | **15.16%** | **Autoencoder** ✅ |
| **ROC AUC** | 0.7457 | 0.5050 | **0.7933** | **Autoencoder** ✅ |

---

## 📈 **Analysis**

### **1. Fixed Windows**

**Original Dataset Wins**:
- F1-Score: **0.80 vs 0.00** (but comprehensive has threshold issue)
- Precision: **0.71 vs 0.00**
- Recall: **0.91 vs 0.00**

**Why Original is Better**:
- Original model was trained and evaluated on larger dataset (3.5M vs 1.69M samples)
- More training data (2.5M vs 1.0M samples)
- More training events (225 vs 176 events)
- Better temporal coverage

**However**: Comprehensive models have excellent ROC AUC (0.9963-1.0), suggesting they can learn well but need threshold optimization.

---

### **2. Sliding Windows (Deployment)**

**Mixed Results**:

**Original Dataset Advantages**:
- ✅ **Higher Precision**: 3.78% vs 2.92% (autoencoder)
- ✅ **Better F1 at high threshold**: 4.80% vs 4.89% (very close)

**Comprehensive Dataset Advantages**:
- ✅ **Higher Recall**: 15.16% vs 6.58% (catches 2.3x more vortices)
- ✅ **Better ROC AUC**: 0.7933 vs 0.7457 (better ranking ability)
- ✅ **Better F1-Score**: 4.89% vs 4.80% (slightly better)

**Trade-off**:
- Original: **Higher precision, lower recall** (fewer false positives, but misses more vortices)
- Comprehensive Autoencoder: **Lower precision, higher recall** (more false positives, but catches more vortices)

---

## 🎯 **Key Insights**

### **1. Dataset Differences Matter**

**Original Dataset (ml_ready_vortex_data.csv)**:
- Larger dataset (3.5M samples)
- More training data (2.5M samples)
- No autoencoder features
- Lower vortex density (0.51% vs 1.05%)

**Comprehensive Dataset (comprehensive_filtered_data_optimized.csv)**:
- Smaller but filtered dataset (1.69M samples)
- Less training data (1.0M samples)
- **Has autoencoder features** (key advantage)
- Higher vortex density (1.05% vs 0.51%)

---

### **2. Autoencoder Features Make a Difference**

**Baseline Model (15 features)**:
- ROC AUC: 0.5050 (essentially random)
- F1-Score: 2.22%
- **Worse than original model**

**Autoencoder Model (19 features)**:
- ROC AUC: 0.7933 (**+57% improvement**)
- F1-Score: 4.89% (**+120% improvement**)
- **Better than original model on ROC AUC and F1**

**Conclusion**: Autoencoder features significantly improve model performance, compensating for smaller dataset.

---

### **3. Why Results Are Different**

1. **Different Training Data**:
   - Original: 2.5M training samples, 225 events
   - Comprehensive: 1.0M training samples, 176 events
   - **Less training data = potentially worse performance**

2. **Different Test Data**:
   - Original: 85,925 sliding windows, 380 positive
   - Comprehensive: 40,354 sliding windows, 442 positive
   - **Different test distribution**

3. **Autoencoder Features**:
   - Original: No autoencoder features available
   - Comprehensive: Autoencoder features available
   - **Autoencoder features compensate for smaller dataset**

---

## ✅ **Final Verdict**

### **Which is Better?**

**For Fixed Windows**:
- **Original dataset wins** (F1=0.80 vs 0.00, but comprehensive has threshold issue)
- However, comprehensive models have perfect ROC AUC (1.0), suggesting they can learn well

**For Sliding Windows (Deployment)**:
- **Comprehensive Autoencoder model wins**:
  - Better ROC AUC: **0.7933 vs 0.7457**
  - Better F1-Score: **4.89% vs 4.80%**
  - Much better Recall: **15.16% vs 6.58%** (catches 2.3x more vortices)
  - Slightly lower Precision: **2.92% vs 3.78%** (more false positives)

**Trade-off**:
- **Original**: Better precision, misses more vortices
- **Comprehensive Autoencoder**: Better recall, catches more vortices, better ranking

---

## 🎯 **Recommendation**

**Use Comprehensive Dataset with Autoencoder Model** because:

1. ✅ **Better ROC AUC** (0.79 vs 0.75) - Excellent ranking ability
2. ✅ **Better Recall** (15% vs 7%) - Catches 2.3x more vortices
3. ✅ **Better F1-Score** (4.89% vs 4.80%)
4. ✅ **Autoencoder features** provide valuable signal
5. ⚠️ **Slightly lower precision** (2.92% vs 3.78%) but acceptable for deployment

**For deployment**: Higher recall is often more important than precision for rare event detection (catching vortices is critical, false positives can be filtered).

---

## 📊 **Summary Table**

| Aspect | Original (ml_ready) | Comprehensive Baseline | Comprehensive Autoencoder |
|--------|---------------------|------------------------|---------------------------|
| **Dataset Size** | 3.5M samples | 1.69M samples | 1.69M samples |
| **Training Samples** | 2.5M | 1.0M | 1.0M |
| **Features** | 15 | 15 | 19 (15 + 4 autoencoder) |
| **Fixed F1** | **0.80** | 0.00* | 0.00* |
| **Sliding F1** | 4.80% | 2.22% | **4.89%** |
| **Sliding Precision** | **3.78%** | 1.28% | 2.92% |
| **Sliding Recall** | 6.58% | 8.37% | **15.16%** |
| **Sliding ROC AUC** | 0.7457 | 0.5050 | **0.7933** |

*Threshold issue - models have excellent ROC AUC but need threshold optimization

---

**Conclusion**: Comprehensive dataset with autoencoder model is **better for deployment** (sliding windows) due to better recall and ROC AUC, despite slightly lower precision.

