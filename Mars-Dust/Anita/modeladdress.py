
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load the labeled sliding windows dataset
# -------------------------------
df = pd.read_csv("address.csv")
print("Dataset shape:", df.shape)

# -------------------------------
# 2. Prepare features and target
# -------------------------------
features = [
    "mean_pressure", 
    "std_pressure", 
    "pressure_change", 
    "pressure_drop_ratio", 
    "z_score",
    "scheme1_detection", 
    "scheme2_detection"
]

# Convert scheme detection columns to integers if necessary
df["scheme1_detection"] = df["scheme1_detection"].astype(int)
df["scheme2_detection"] = df["scheme2_detection"].astype(int)

# The target column: we assume ml_label indicates a dust devil event (True/1)
target = "ml_label"

# Drop any rows with missing values in the features or target
df = df.dropna(subset=features + [target])

X = df[features]
y = df[target].astype(int)  # Convert True/False to 1/0

# -------------------------------
# 3. Create a time-aware train/test split
# -------------------------------
# For time series data, we use a chronological split
split_index = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# -------------------------------
# 4. Train a RandomForestClassifier
# -------------------------------
# Option 1: Adjust class weights to reduce false positives:
#   class_weight="balanced" or a custom dict like {0: 1, 1: 0.5}
#   to penalize over-predicting the positive class.

# Example: Balanced approach:
# rf_model = RandomForestClassifier(
#     n_estimators=100, 
#     random_state=42,
#     class_weight="balanced"
# )

# If you prefer not to use class weights, comment out the above and use:
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

# -------------------------------
# 5. Evaluate the model on the test set (Default Threshold = 0.5)
# -------------------------------
print("\n--- Evaluation at Default Threshold = 0.5 ---")
y_pred_default = rf_model.predict(X_test)

acc_def = accuracy_score(y_test, y_pred_default)
f1_def = f1_score(y_test, y_pred_default)
prec_def = precision_score(y_test, y_pred_default)
rec_def = recall_score(y_test, y_pred_default)

print("Accuracy:", acc_def)
print("Precision:", prec_def)
print("Recall:", rec_def)
print("F1 Score:", f1_def)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_default))

# -------------------------------
# 6. Probability & Threshold Tuning
# -------------------------------
# Get the probability of the positive class
y_prob = rf_model.predict_proba(X_test)[:, 1]

# We'll search for a threshold that improves precision (thus reducing false positives)
thresholds = np.linspace(0, 1, 101)
best_f1 = 0
best_thresh = 0
best_precision = 0
best_recall = 0

f1_scores = []
precisions = []
recalls = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1_t = f1_score(y_test, y_pred_t, zero_division=0)
    prec_t = precision_score(y_test, y_pred_t, zero_division=0)
    rec_t = recall_score(y_test, y_pred_t, zero_division=0)
    
    f1_scores.append(f1_t)
    precisions.append(prec_t)
    recalls.append(rec_t)
    
    if f1_t > best_f1:
        best_f1 = f1_t
        best_thresh = t
        best_precision = prec_t
        best_recall = rec_t

print("\n--- Threshold Tuning Results ---")
print(f"Best Threshold: {best_thresh:.2f}")
print(f"F1 at Best Threshold: {best_f1:.3f}")
print(f"Precision at Best Threshold: {best_precision:.3f}")
print(f"Recall at Best Threshold: {best_recall:.3f}")

# -------------------------------
# 7. Plot Precision, Recall, F1 vs. Threshold
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, label="Precision", marker="x")
plt.plot(thresholds, recalls, label="Recall", marker="s")
plt.plot(thresholds, f1_scores, label="F1", marker="o")
plt.axvline(best_thresh, color='gray', linestyle='--', label=f"Best Threshold: {best_thresh:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Metric Value")
plt.title("Precision, Recall, F1 vs. Threshold")
plt.legend()
plt.grid(True)
plt.show()
