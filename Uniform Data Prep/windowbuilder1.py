import os
import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\data\dataset_index\dataset_index_2004.csv"
PREPROCESS_DIR = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\dataset_index\preprocess"
LABEL_DIR = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\processed\labels_2004"
OUT_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\data\dataset_index\windows_2004"
# =================

WINDOW = 128
STRIDE_FAR = 16
STRIDE_NEAR = 4

# "Near crossing" radius in timesteps (≈10 minutes if ~8s cadence → ~75)
NEAR_RADIUS = 75

os.makedirs(OUT_DIR, exist_ok=True)

def make_windows_adaptive(X, y):
    n = X.shape[0]
    half = WINDOW // 2
    X_out, y_out = [], []

    # mark indices near any positive label
    pos_idx = np.where(y == 1)[0]
    near = np.zeros(n, dtype=bool)
    for p in pos_idx:
        lo = max(0, p - NEAR_RADIUS)
        hi = min(n, p + NEAR_RADIUS + 1)
        near[lo:hi] = True

    start = 0
    while start + WINDOW <= n:
        center = start + half
        stride = STRIDE_NEAR if near[center] else STRIDE_FAR

        X_out.append(X[start:start + WINDOW])
        y_out.append(int(y[center] == 1))

        start += stride

    return np.asarray(X_out, np.float32), np.asarray(y_out, np.uint8)

df = pd.read_csv(INDEX_CSV)
df = df[df["exclude"] == False].reset_index(drop=True)

splits_X = {"train": [], "val": [], "test": []}
splits_y = {"train": [], "val": [], "test": []}

processed = 0
missing = 0
mismatch = 0

for _, row in df.iterrows():
    file_key = row["file_key"]
    split = row["split"]

    x_path = os.path.join(PREPROCESS_DIR, f"{file_key}_X_63.npy")
    y_path = os.path.join(LABEL_DIR, f"{file_key}_y.npy")

    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        missing += 1
        continue

    X = np.load(x_path)
    y = np.load(y_path)

    if X.shape[0] != y.shape[0]:
        mismatch += 1
        continue

    Xw, yw = make_windows_adaptive(X, y)
    if Xw.shape[0] == 0:
        continue

    splits_X[split].append(Xw)
    splits_y[split].append(yw)

    processed += 1
    if processed % 100 == 0:
        print(f"Processed {processed} files...")

for split in ["train", "val", "test"]:
    if splits_X[split]:
        X_all = np.concatenate(splits_X[split], axis=0)
        y_all = np.concatenate(splits_y[split], axis=0)
    else:
        X_all = np.empty((0, WINDOW, 63), dtype=np.float32)
        y_all = np.empty((0,), dtype=np.uint8)

    np.save(os.path.join(OUT_DIR, f"{split}_X.npy"), X_all)
    np.save(os.path.join(OUT_DIR, f"{split}_y.npy"), y_all)

    print(f"\n{split.upper()}:")
    print("  windows:", X_all.shape[0])
    if X_all.shape[0] > 0:
        print("  positive windows:", int(y_all.sum()))
        print("  positive fraction:", float(y_all.mean()))

print("\nDone.")
print("Processed files:", processed)
print("Missing X or y:", missing)
print("Mismatched length:", mismatch)
print("Saved to:", OUT_DIR)