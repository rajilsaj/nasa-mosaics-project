# Temporal Split Files: What They Do and Why They Matter

## 📁 **File Overview**

### **ML Dataset Files** (Pressure Time-Series Data)

#### **1. `ml_train.csv`** ⭐⭐⭐ **CRITICAL**
**What it contains:**
- ~1.02M pressure measurements (60% of data)
- Early time period (first 60% chronologically)
- All columns: SCLK, PRESSURE, sol, time, gt_detection_win, autoencoder features, etc.
- Time range: SCLK 667042464 to 672352720

**What it's used for:**
- **Training the Random Forest model**
- Learning patterns from pressure data
- Feature engineering (creating your 15 features)
- Negative sampling (balancing to 1:1 ratio)

**Why it's critical:**
- This is what the model **learns from**
- Quality of training data = quality of model
- Must be chronologically first (no future data leakage)

**When you use it:**
- Feature engineering → `train_features.csv`
- Model training → `rf_model.pkl`
- Baseline establishment

---

#### **2. `ml_val.csv`** ⭐⭐⭐ **CRITICAL**
**What it contains:**
- ~254K pressure measurements (15% of data)
- Middle time period (between train and test)
- Same columns as train
- Time range: SCLK 672364466 to 673159782
- **Gap from train**: 11,746 SCLK units (temporal isolation)

**What it's used for:**
- **Hyperparameter tuning** (finding best RF parameters)
- **Threshold optimization** (finding best decision threshold)
- **Model selection** (comparing different models)
- **Early stopping** (preventing overfitting)

**Why it's critical:**
- **Prevents overfitting** - tests model on unseen data
- **Tuning without touching test set** - keeps test pristine
- **Realistic evaluation** - natural imbalance (not balanced like train)

**When you use it:**
- Feature engineering → `val_features.csv`
- Model validation during training
- Threshold optimization
- Probability adjustment validation

---

#### **3. `ml_test.csv`** ⭐⭐⭐ **CRITICAL**
**What it contains:**
- ~407K pressure measurements (24% of data)
- Latest time period (final 24% chronologically)
- Same columns as train/val
- Time range: SCLK 673178500 to 674883815
- **Gap from val**: 18,718 SCLK units (temporal isolation)

**What it's used for:**
- **Final model evaluation** (only after all tuning is done)
- **Performance reporting** (the numbers you publish)
- **Deployment simulation** (real-world scenario)
- **Never touch until final evaluation!**

**Why it's critical:**
- **Unbiased evaluation** - model has never seen this data
- **Realistic deployment scenario** - latest data = future predictions
- **One-time use** - only evaluate once, or results are invalid

**When you use it:**
- Final evaluation only (after all tuning)
- Performance metrics (precision, recall, F1)
- Publication results

---

### **Jackson Event Files** (Ground Truth Vortex Events)

#### **4. `jackson_train.csv`** ⭐⭐ **IMPORTANT**
**What it contains:**
- 177 vortex event locations (SCLK timestamps)
- Events that occur in the training time period
- Ground truth labels (where vortices actually happened)

**What it's used for:**
- **Window extraction** - find 60-sample windows before each event
- **Positive window labeling** - mark windows that contain vortices
- **Training label creation** - create positive examples

**Why it's important:**
- Tells you **where vortices actually occurred** in training period
- Used to extract positive windows (60 samples before each event)
- Creates ground truth for model to learn from

**When you use it:**
- Window extraction (already done)
- Creating positive training examples
- Validating window extraction

---

#### **5. `jackson_val.csv`** ⭐⭐ **IMPORTANT**
**What it contains:**
- 48 vortex event locations
- Events in validation time period
- Ground truth for validation set

**What it's used for:**
- **Validation window extraction** (already done)
- **Validation label creation** - create positive examples for validation
- **Model evaluation** - compare predictions to ground truth

**Why it's important:**
- Provides ground truth for validation set
- Used to evaluate model performance during tuning
- Creates validation labels (positive/negative windows)

**When you use it:**
- Window extraction (already done)
- Validation evaluation
- Threshold optimization

---

#### **6. `jackson_test.csv`** ⭐⭐ **IMPORTANT**
**What it contains:**
- 80 vortex event locations
- Events in test time period
- Ground truth for final evaluation

**What it's used for:**
- **Test window extraction** (already done)
- **Final evaluation labels** - compare final predictions to ground truth
- **Performance metrics** - calculate precision, recall, F1

**Why it's important:**
- Provides ground truth for test set
- Used for final model evaluation
- Creates test labels (positive/negative windows)

**When you use it:**
- Window extraction (already done)
- Final evaluation only
- Performance reporting

---

## 🔄 **How They Work Together**

### **Pipeline Flow:**

```
1. ml_train.csv + jackson_train.csv
   ↓
   Extract windows → train_windows.csv
   ↓
   Feature engineering → train_features.csv
   ↓
   Negative sampling → train_balanced.csv
   ↓
   Train RF model

2. ml_val.csv + jackson_val.csv
   ↓
   Extract windows → val_windows.csv
   ↓
   Feature engineering → val_features.csv
   ↓
   Validate model (tune hyperparameters, optimize threshold)

3. ml_test.csv + jackson_test.csv
   ↓
   Extract windows → test_windows.csv
   ↓
   Feature engineering → test_features.csv
   ↓
   Final evaluation (ONLY after all tuning is done!)
```

---

## ⚠️ **Critical Rules**

### **Temporal Order (MUST FOLLOW)**
1. **Train** comes first (early time period)
2. **Gap** (no data, prevents leakage)
3. **Validation** comes second (middle time period)
4. **Gap** (no data, prevents leakage)
5. **Test** comes last (late time period)

**Why**: Model learns from past, validates on present, tests on future.

### **Never Mix Splits**
- ❌ Don't use test data for training
- ❌ Don't use test data for validation
- ❌ Don't tune on test set
- ✅ Train on train, tune on val, test on test

### **Test Set Rules**
- ✅ Only evaluate once (after all tuning)
- ✅ Never use for hyperparameter tuning
- ✅ Never use for feature selection
- ✅ Keep it pristine until final evaluation

---

## 📊 **Importance Ranking**

### **Most Critical (⭐⭐⭐)**
1. **ml_train.csv** - Model learns from this
2. **ml_val.csv** - Prevents overfitting, tunes model
3. **ml_test.csv** - Final evaluation, performance reporting

### **Important (⭐⭐)**
4. **jackson_train.csv** - Creates training labels
5. **jackson_val.csv** - Creates validation labels
6. **jackson_test.csv** - Creates test labels

---

## 🎯 **What You'll Use Next**

### **For Feature Engineering:**
- `ml_train.csv` → Create `train_features.csv`
- `ml_val.csv` → Create `val_features.csv`
- `ml_test.csv` → Create `test_features.csv`

### **For Model Training:**
- `train_features.csv` → Train RF model
- `val_features.csv` → Validate and tune
- `test_features.csv` → Final evaluation (later!)

### **For Window Extraction:**
- Already done! Windows are in `data/windows/`
- But Jackson files were used to find where to extract windows

---

## 💡 **Key Takeaways**

1. **ML files** = Pressure time-series data (what model sees)
2. **Jackson files** = Ground truth events (where vortices happened)
3. **Train** = Learn patterns
4. **Val** = Tune and validate
5. **Test** = Final evaluation (use once!)
6. **Temporal order** = Critical (no future data leakage)
7. **Gaps** = Prevent data leakage between splits

**All files are important, but ML files are what the model actually uses for training/evaluation!**

