import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

# Load your CSV (adjust path as needed)
df = pd.read_csv("data/ml_ready_vortex_data.csv")


# Drop rows with missing pressure or labels
df.dropna(subset=["PRESSURE", "PRESSURE_MA_500", "gt_detection_win"], inplace=True)

# Calculate pressure difference and % drop
df["pressure_diff"] = df["PRESSURE"] - df["PRESSURE_MA_500"]
df["pressure_diff_percent"] = (df["pressure_diff"] / df["PRESSURE_MA_500"]) * 100

features = ["pressure_diff", "pressure_diff_percent"]
X = df[features]
y = df["gt_detection_win"]  # Label: True if inside detection window

# Compute class imbalance ratio
ratio = (y == 0).sum() / (y == 1).sum()
print(f"Class imbalance ratio (False:True): {ratio:.2f}")


# Use chronological split — no shuffle!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # handle imbalance automatically
    random_state=42
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Get predicted probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# Compute precision-recall
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], label='Precision')
plt.plot(thresholds, recall[:-1], label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision & Recall vs Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Pick a custom threshold (e.g., 0.3) from the curve
custom_threshold = 0.3
y_pred_custom = (y_scores >= custom_threshold).astype(int)

print("\nCustom Threshold Report (Threshold = 0.3):")
print(classification_report(y_test, y_pred_custom))
