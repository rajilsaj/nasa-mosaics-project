import os

import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\dataset_index_2004.csv"
PREPROCESS_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\preprocess"
LABEL_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\labels_2004_2"
OUT_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\windows_2004"
# =================

WINDOW = 128
STRIDE_FAR = 16
STRIDE_NEAR = 4
NEAR_RADIUS = 75


def to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def make_windows_adaptive(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    half = WINDOW // 2
    X_out, y_out = [], []

    pos_idx = np.where(y > 0)[0]
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
        y_out.append(int(y[center]))
        start += stride

    if len(X_out) == 0:
        return np.empty((0, WINDOW, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.uint8)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INDEX_CSV)
    df = df.loc[~df["exclude"].apply(to_bool)].reset_index(drop=True)

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
            print(f"Length mismatch for {file_key}: X={X.shape[0]} y={y.shape[0]}")
            continue

        Xw, yw = make_windows_adaptive(X, y)
        if len(Xw) == 0:
            continue

        splits_X[split].append(Xw)
        splits_y[split].append(yw)
        processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed} files...")

    for split in ("train", "val", "test"):
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
        print("  positives:", int((y_all > 0).sum()))
        print("  positive fraction:", float((y_all > 0).mean()) if len(y_all) else 0.0)
        if len(y_all):
            unique, counts = np.unique(y_all, return_counts=True)
            print("  class counts:", dict(zip(unique.tolist(), counts.tolist())))

    print("\nDone.")
    print("Processed files:", processed)
    print("Missing X or y:", missing)
    print("Mismatched length:", mismatch)
    print("Saved windows to:", OUT_DIR)


if __name__ == "__main__":
    main()
