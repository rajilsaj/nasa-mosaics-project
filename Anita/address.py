
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# -------------------------------
# Parameters for fixed and sliding windows:
# -------------------------------
FIXED_BEFORE = 100    # number of rows before the matching index (historical data only)
SUB_WINDOW_SIZE = 10  # number of rows per sliding (sub) window
STEP_SIZE = 1         # slide one row at a time

# -------------------------------
# Detection scheme thresholds (default values):
# -------------------------------
DEFAULT_SIMPLE_PRESSURE_DROP_THRESHOLD = 0.01  # 1% drop: scheme 1 threshold (relative drop)
EXPERT_Z_THRESHOLD = 1.0                       # scheme 2 threshold: tunable z-score threshold

# -------------------------------
# 1. Read the vortex CSV file and extract all SCLK values
# -------------------------------
vortex_df = pd.read_csv("Jackson_vortex_detections_reformatted_augmented.csv")
vortex_sclk_list = vortex_df["SCLK"].tolist()
print(f"Found {len(vortex_sclk_list)} SCLK values in the vortex file.")

# -------------------------------
# 2. Read the ml CSV file
# -------------------------------
ml_df = pd.read_csv("ml_ready_vortex_data.csv")

# --- Ensure required columns exist ---
required_cols = ["SCLK", "PRESSURE", "gt_detection_win", "gt_fwhm"]
for col in required_cols:
    if col not in ml_df.columns:
        raise ValueError(f"Required column '{col}' not found in ml_df.")

# --- Convert gt_detection_win and gt_fwhm to boolean if they are strings ---
ml_df["gt_detection_win"] = ml_df["gt_detection_win"].astype(bool)
ml_df["gt_fwhm"] = ml_df["gt_fwhm"].astype(bool)

# Prepare a list to store labeled windows for all vortex SCLK values:
all_labeled_windows = []

# -------------------------------
# 3. Loop through each vortex SCLK value and process:
# -------------------------------
for vortex_sclk in vortex_sclk_list:
    # Locate the row in ml_df where SCLK equals the current vortex SCLK
    matching_rows = ml_df[ml_df["SCLK"] == vortex_sclk]
    if matching_rows.empty:
        print(f"SCLK value {vortex_sclk} not found in ml data; skipping.")
        continue
    matching_index = matching_rows.index[0]
    print(f"Processing vortex SCLK {vortex_sclk} found at ml index: {matching_index}")
    
    # Define the fixed window around the matching index:
    fixed_start_index = max(matching_index - FIXED_BEFORE, 0)
    fixed_end_index = matching_index - 1  # Exclude the event row and any future data
    print(f"Extracting fixed window for SCLK {vortex_sclk} from index {fixed_start_index} to {fixed_end_index}")
    
    fixed_window_df = ml_df.iloc[fixed_start_index: fixed_end_index + 1]
    
    # Slide a sub-window over the fixed window.
    for i in range(0, len(fixed_window_df) - SUB_WINDOW_SIZE + 1, STEP_SIZE):
        sub_window = fixed_window_df.iloc[i: i + SUB_WINDOW_SIZE]
        
        # Use the right-hand side row (last row) of the sub-window for labeling.
        r = sub_window.iloc[-1]
        
        # Scheme 3 (ML Labeling) logic using ground truth:
        if r["gt_detection_win"]:
            ml_label = True
        elif (not r["gt_detection_win"]) and (not r["gt_fwhm"]):
            ml_label = False
        elif (not r["gt_detection_win"]) and (r["gt_fwhm"]):
            continue  # omit this window as it is already too late
        else:
            ml_label = False

        # Compute features from the 'PRESSURE' column for the sub-window:
        initial_pressure = sub_window["PRESSURE"].iloc[0]
        final_pressure = sub_window["PRESSURE"].iloc[-1]
        mean_pressure = sub_window["PRESSURE"].mean()
        std_pressure = sub_window["PRESSURE"].std()
        pressure_change = final_pressure - initial_pressure
        pressure_drop_ratio = (initial_pressure - final_pressure) / initial_pressure
        
        # Scheme 1: Simple threshold detection based on relative pressure drop.
        scheme1_detection = pressure_drop_ratio >= DEFAULT_SIMPLE_PRESSURE_DROP_THRESHOLD
        
        # Scheme 2: Expert system detection using z-score.
        if std_pressure > 0:
            z_score = (initial_pressure - final_pressure) / std_pressure
        else:
            z_score = 0
        scheme2_detection = z_score >= EXPERT_Z_THRESHOLD

        # -------------------------------
        # Feature Engineering from Multiple Time Scales
        # -------------------------------
        # Define a long-term window: data from the start of the fixed window up to the end of the current sub-window.
        long_term_window = fixed_window_df.iloc[:i + SUB_WINDOW_SIZE]
        long_term_mean = long_term_window["PRESSURE"].mean()
        long_term_std = long_term_window["PRESSURE"].std()
        # Compute the exponential moving average (EMA) of pressure over the long-term window.
        # The span is set to half the length of the long-term window, with a minimum of 1 to avoid division by zero.
        span_val = max(len(long_term_window) // 2, 1)
        ema_pressure = long_term_window["PRESSURE"].ewm(span=span_val, adjust=False).mean().iloc[-1]
        # Trend feature: difference between the sub-window mean and the long-term mean.
        trend = mean_pressure - long_term_mean

        # -------------------------------
        # Gather the data for this sub-window:
        # -------------------------------
        row_data = r.to_dict()
        row_data["sub_window_start_index"] = sub_window.index[0]
        row_data["sub_window_end_index"] = sub_window.index[-1]
        row_data["mean_pressure"] = mean_pressure
        row_data["std_pressure"] = std_pressure
        row_data["pressure_change"] = pressure_change
        row_data["pressure_drop_ratio"] = pressure_drop_ratio
        row_data["z_score"] = z_score
        row_data["scheme1_detection"] = scheme1_detection
        row_data["scheme2_detection"] = scheme2_detection
        row_data["ml_label"] = ml_label
        row_data["vortex_sclk"] = vortex_sclk

        # Add new features from multiple time scales:
        row_data["long_term_mean"] = long_term_mean
        row_data["long_term_std"] = long_term_std
        row_data["ema_pressure"] = ema_pressure
        row_data["trend"] = trend

        all_labeled_windows.append(row_data)

# -------------------------------
# 4. Create a DataFrame from all the labeled sliding windows and save to CSV
# -------------------------------
labeled_df = pd.DataFrame(all_labeled_windows)
labeled_df.to_csv("address.csv", index=False)

print("Labeled sliding windows saved to 'address.csv'.")
print("Number of labeled windows:", len(labeled_df))

# ------------------------------------------------------------------------------
# 5. Threshold Tuning for Scheme 1:
#    We use the continuous 'pressure_drop_ratio' as the score and tune the threshold
#    to maximize the F1 score (and report precision and recall) compared to ml_label.
# ------------------------------------------------------------------------------
y_true = labeled_df["ml_label"].astype(int).values
y_scores = labeled_df["pressure_drop_ratio"].values

# Try thresholds across the range of observed pressure_drop_ratio values.
thresholds = np.linspace(y_scores.min(), y_scores.max(), 101)
f1_scores = []
precisions = []
recalls = []

best_f1 = 0
best_thresh = None

for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1_scores.append(f1)
    precisions.append(prec)
    recalls.append(rec)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

# Calculate metrics at the best threshold
y_pred_best = (y_scores >= best_thresh).astype(int)
best_precision = precision_score(y_true, y_pred_best, zero_division=0)
best_recall = recall_score(y_true, y_pred_best, zero_division=0)

print("\nThreshold Tuning for Scheme 1 (Pressure Drop Ratio):")
print(f"Best threshold: {best_thresh:.4f}")
print(f"F1 Score at best threshold: {best_f1:.3f}")
print(f"Precision at best threshold: {best_precision:.3f}")
print(f"Recall at best threshold: {best_recall:.3f}")

# ------------------------------------------------------------------------------
# 6. Plot the metrics versus threshold values:
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label="F1 Score", marker="o")
plt.plot(thresholds, precisions, label="Precision", marker="x")
plt.plot(thresholds, recalls, label="Recall", marker="s")
plt.axvline(best_thresh, color='gray', linestyle='--', label=f"Best Threshold: {best_thresh:.4f}")
plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Threshold Tuning for Scheme 1 (Pressure Drop Ratio)")
plt.legend()
plt.grid(True)
plt.show()

