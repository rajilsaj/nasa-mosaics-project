import numpy as np
import joblib
import os

# Path to your scaled dataset
DATA_DIR = r"C:\Users\roger\OneDrive\Desktop\Cassini New\Spacecraft Data\dataset_index\windows_2004\scaled"

train_y_path = os.path.join(DATA_DIR, "train_y.npy")

# Load labels
y = np.load(train_y_path)

pos = int(y.sum())
neg = int(len(y) - pos)

# Compute weight
pos_weight = neg / pos

class_weights = {
    "negative": 1.0,
    "positive": float(pos_weight)
}

print("Negative samples:", neg)
print("Positive samples:", pos)
print("Positive class weight:", pos_weight)

# Save weights
save_path = os.path.join(DATA_DIR, "class_weights.pkl")
joblib.dump(class_weights, save_path)

print("Saved class weights to:", save_path)