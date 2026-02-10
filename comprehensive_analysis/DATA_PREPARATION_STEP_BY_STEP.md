# Data Preparation: Step-by-Step Breakdown
## How the Data Preparation Pipeline Works

---

## 🎯 **Overview**

The data preparation process takes the comprehensive dataset and prepares it for Random Forest training by:
1. Loading and validating data
2. Creating temporal splits (train/val/test) with gaps
3. Extracting 60-sample windows before each vortex event
4. Preserving all features including autoencoder

---

## 📋 **STEP 1: Load and Validate Data**

### **1.1 Load Comprehensive Dataset**
```python
ml_df = pd.read_csv("comprehensive_filtered_data_optimized.csv")
```
**What happens:**
- Loads 1,694,934 pressure measurements
- Includes all 11 columns (PRESSURE, autoencoder features, etc.)
- Checks for autoencoder features presence

**Output**: DataFrame with all pressure data

---

### **1.2 Load Jackson Events**
```python
jackson_df = pd.read_csv("Jackson_vortex_detections_reformatted_augmented.csv")
```
**What happens:**
- Loads 309 vortex event locations (SCLK timestamps)
- These are ground truth labels (where vortices actually happened)

**Output**: DataFrame with vortex event SCLK timestamps

---

### **1.3 Data Validation**
**What happens:**
- Convert SCLK to numeric (handle any text values)
- Remove rows with invalid SCLK values
- **Sort by SCLK** (critical for temporal order!)
- Check SCLK ranges overlap between datasets

**Why**: Ensures data is clean and temporally ordered

**Output**: Cleaned, sorted DataFrames

---

## 📊 **STEP 2: Temporal Splitting**

### **2.1 Calculate Split Indices**
```python
n = len(ml_df)  # Total samples: 1,694,934

train_end_idx = int(n * 0.60)        # 60% = 1,016,960
val_start_idx = int(n * 0.605)       # 60% + 0.5% gap
val_end_idx = int(n * 0.755)         # 60% + 0.5% + 15%
test_start_idx = int(n * 0.76)       # 60% + 0.5% + 15% + 0.5% gap
```

**Visual Representation:**
```
[========== TRAIN ==========][GAP][=== VAL ===][GAP][==== TEST ====]
0                          60%  60.5%       75.5% 76%           100%
1,016,960 samples           Gap  254,240      Gap  406,785 samples
```

---

### **2.2 Split ML Data (Pressure Time-Series)**
```python
ml_train = ml_df.iloc[:train_end_idx].copy()      # First 60%
ml_val = ml_df.iloc[val_start_idx:val_end_idx].copy()  # Middle 15%
ml_test = ml_df.iloc[test_start_idx:].copy()     # Last 24.5%
```

**What happens:**
- **Train**: Rows 0 to 1,016,960 (early time period)
- **Gap 1**: Rows 1,016,960 to 1,025,435 (skipped, prevents leakage)
- **Val**: Rows 1,025,435 to 1,279,675 (middle time period)
- **Gap 2**: Rows 1,279,675 to 1,288,149 (skipped, prevents leakage)
- **Test**: Rows 1,288,149 to end (late time period)

**Why gaps?**: Prevents data leakage - model can't "peek" into future

---

### **2.3 Split Jackson Events (Ground Truth)**
```python
# Find SCLK ranges for each split
train_sclk_max = ml_train['SCLK'].max()
val_sclk_min = ml_val['SCLK'].min()
val_sclk_max = ml_val['SCLK'].max()
test_sclk_min = ml_test['SCLK'].min()

# Filter Jackson events by SCLK range
jackson_train = jackson_df[jackson_df['SCLK'] <= train_sclk_max]
jackson_val = jackson_df[(jackson_df['SCLK'] >= val_sclk_min) & 
                         (jackson_df['SCLK'] <= val_sclk_max)]
jackson_test = jackson_df[jackson_df['SCLK'] >= test_sclk_min]
```

**What happens:**
- Jackson events are filtered to match ML split time periods
- Train: 177 events (in training time period)
- Val: 48 events (in validation time period)
- Test: 80 events (in test time period)

**Why**: Each split needs its own ground truth events

---

### **2.4 Save Splits**
```python
ml_train.to_csv("data/splits/ml_train.csv")
ml_val.to_csv("data/splits/ml_val.csv")
ml_test.to_csv("data/splits/ml_test.csv")
# ... same for Jackson files
```

**Output**: 6 CSV files in `data/splits/`

---

### **2.5 Validate Temporal Isolation**
```python
gap1_size = val_min - train_max  # Should be > 0
gap2_size = test_min - val_max   # Should be > 0
```

**What happens:**
- Checks that gaps exist (no overlap)
- Verifies no SCLK values appear in multiple splits

**Why**: Ensures no data leakage between splits

---

## 🔍 **STEP 3: Window Extraction**

### **3.1 For Each Jackson Event**

**Input**: One vortex event with SCLK timestamp

**Process**:

#### **Step 3.1.1: Find Event Location**
```python
target_sclk = row['SCLK']  # e.g., 672352720
matches = ml_df[ml_df['SCLK'] == target_sclk]
target_position = matches.index[0]  # Position in ML dataframe
```

**What happens:**
- Find where this vortex event occurs in the pressure data
- Get the row index (position) in the ML dataframe

---

#### **Step 3.1.2: Find Precursor Region**
```python
# Look backward from event position
search_space = ml_df.iloc[:target_position + 1]
precursor_region = search_space[search_space['gt_detection_win'] == True]
first_precursor_position = precursor_region.index[0]
```

**What happens:**
- Search backward from event to find where `gt_detection_win == True`
- This is the "precursor region" (pressure drop before vortex)
- Find the FIRST occurrence (start of precursor)

**Visual**:
```
[Window: 60 samples] [Precursor starts] [Vortex Event]
     ← backward          gt_detection_win=True    SCLK
```

---

#### **Step 3.1.3: Extract 60-Sample Window**
```python
start_position = max(0, first_precursor_position - 60)
end_position = first_precursor_position - 1
window = ml_df.iloc[start_position:end_position + 1]
```

**What happens:**
- Go back 60 samples from where precursor starts
- Extract those 60 samples
- Window ends just BEFORE precursor starts

**Example**:
```
If precursor starts at position 1000:
- Start: 1000 - 60 = 940
- End: 1000 - 1 = 999
- Window: rows 940 to 999 (60 samples)
```

---

#### **Step 3.1.4: Add Metadata**
```python
window['window_id'] = successful_count  # Unique ID
window['event_sclk'] = target_sclk      # Original event SCLK
window['split'] = split_name            # 'train', 'val', or 'test'
window['label'] = True                  # Positive window
```

**What happens:**
- Adds tracking information to each window
- Labels window as positive (contains vortex)

---

### **3.2 Repeat for All Events**

**Process**:
- Loop through all Jackson events in each split
- Extract window for each event
- Collect all windows

**Results**:
- Train: 176 windows (from 177 events, 1 failed)
- Val: 48 windows (from 48 events, all succeeded)
- Test: 80 windows (from 80 events, all succeeded)

---

### **3.3 Save Windows**
```python
all_windows_df = pd.concat(all_windows)
all_windows_df.to_csv("data/windows/train_windows.csv")
```

**What happens:**
- Combine all windows into one DataFrame
- Each window is 60 rows (60 samples)
- Total: 176 windows × 60 = 10,560 rows for train

**Output**: 3 CSV files in `data/windows/`

---

## 🔑 **Key Concepts Explained**

### **Why 60 Samples Backward?**
- Vortex events have **precursor pressure drops** before they occur
- Model learns to detect these pressure patterns
- 60 samples = enough to capture the pressure drop pattern

### **Why Gaps Between Splits?**
- **Prevents data leakage**: Model can't use future information
- **Realistic evaluation**: Simulates real-world deployment
- **Temporal causality**: Model learns from past, predicts future

### **Why Extract Windows?**
- RF needs **fixed-size inputs** (60 features per window)
- Windows capture **temporal patterns** (pressure changes over time)
- Each window = one training example

### **What Gets Preserved?**
- ✅ All original columns (PRESSURE, SCLK, sol, time, etc.)
- ✅ Autoencoder features (autoencoder_window_hits, autoencoder_positive_hit)
- ✅ Ground truth labels (gt_detection_win)
- ✅ All metadata

---

## 📊 **Data Flow Summary**

```
comprehensive_filtered_data_optimized.csv (1.69M rows)
    ↓
[Sort by SCLK]
    ↓
[Temporal Split]
    ├── ml_train.csv (1.02M rows, 60%)
    ├── ml_val.csv (254K rows, 15%)
    └── ml_test.csv (407K rows, 24.5%)
    ↓
[Window Extraction - for each Jackson event]
    ├── train_windows.csv (176 windows × 60 = 10,560 rows)
    ├── val_windows.csv (48 windows × 60 = 2,880 rows)
    └── test_windows.csv (80 windows × 60 = 4,800 rows)
    ↓
[Ready for Feature Engineering]
```

---

## ✅ **What We Get**

### **Temporal Splits** (`data/splits/`)
- 6 files: ml_train/val/test.csv + jackson_train/val/test.csv
- Clean, temporally isolated data
- Ready for feature engineering

### **Extracted Windows** (`data/windows/`)
- 3 files: train/val/test_windows.csv
- Each window = 60 samples before vortex event
- All features preserved (including autoencoder)
- Labeled as positive (True)

---

## 🎯 **Next Steps**

Windows are now ready for:
1. **Feature Engineering**: Create 15 features + autoencoder features
2. **Negative Sampling**: Add negative windows (non-vortex) for training
3. **Model Training**: Train Random Forest on engineered features

---

## 💡 **Key Takeaways**

1. **Temporal order is critical** - data must be sorted by SCLK
2. **Gaps prevent leakage** - no overlap between splits
3. **Windows capture patterns** - 60 samples before each event
4. **All features preserved** - including autoencoder features
5. **Each split is independent** - train/val/test don't overlap

**The data is now ready for feature engineering!**

