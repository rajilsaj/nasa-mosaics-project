import os
import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"/home/jhuss/nasa-mosaics-project/data/dataset_index/dataset_index_2004-2012.csv"
PREPROCESS_DIR = r"/home/jhuss/nasa-mosaics-project/data/dataset_index/preprocess_datasetB"
LABEL_DIR = r"/home/jhuss/nasa-mosaics-project/data/dataset_index/labels_B"
OUT_DIR = r"/home/jhuss/nasa-mosaics-project/data/dataset_index/windows_2004-2012_v2_B"
# =================

WINDOW = 128
STRIDE = 16                               
STRIDE_FAR = 32
N_NOISE_BEFORE = 8                        #was 2
NEAR_RADIUS = 64                          #was 40
N_FEATURES = 63

def to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def window_label(y_slice: np.ndarray) -> int:
    # adds the ability to label a window true if any crossing event happens in it
    return int((y_slice > 0).any())


def make_windows_adaptive(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build windows anchored to crossing events.

    Returns X_out shape (N, WINDOW, n_features) and y_out shape (N,).
    """
    n = X.shape[0]
    X_out, y_out = [], []

    pos = (y > 0).astype(np.uint8)
    padded = np.concatenate([[0], pos, [0]])
    ends   = np.where(np.diff(padded) == -1)[0]

    for event_end in ends:
        win_start = event_end - WINDOW

        if win_start < 0:
            continue

        X_out.append(X[win_start:win_start + WINDOW])
        y_out.append(window_label(y[win_start:win_start + WINDOW]))
        
        for i in range(1, N_NOISE_BEFORE + 1):
            noise_end = win_start - (i-1) * WINDOW
            noise_start = noise_end - WINDOW
            if noise_start < 0:
                break
            X_out.append(X[noise_start:noise_start + WINDOW])
            y_out.append(window_label(y[noise_start:noise_start + WINDOW]))

    near = np.zeros(n, dtype=bool)
    for p in np.where(y > 0)[0]:
        lo = max(0, p - NEAR_RADIUS - WINDOW)
        hi = min(n, p + NEAR_RADIUS + 1)
        near[lo:hi] = True

    start = 0
    while start + WINDOW <= n:
        win_end = start + WINDOW - 1    # uses last timestamp in the window
        lbl = window_label(y[start:start + WINDOW])
        if near[win_end] and lbl == 0:
            X_out.append(X[start:start + WINDOW])
            y_out.append(0)

        start += STRIDE

    if len(X_out) == 0:
        return np.empty((0, WINDOW, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.uint8)


def count_windows_for_file(X: np.ndarray, y: np.ndarray) -> int:
    """
    Dry-run of the window loop — counts how many windows a file will produce
    without storing anything. Used in Pass 1 to pre-allocate disk space.
    """
    n = X.shape[0]
    count = 0

    pos = (y > 0).astype(np.uint8)
    padded = np.concatenate([[0], pos, [0]])
    ends = np.where(np.diff(padded) == -1)[0]

    for event_end in ends:
        win_start = event_end - WINDOW
        if win_start < 0:
            continue
        count += 1
        
        for i in range(1, N_NOISE_BEFORE + 1):
            noise_end = win_start - (i - 1) * WINDOW
            noise_start = noise_end - WINDOW
            if noise_start < 0:
                break
            count += 1

    near = np.zeros(n, dtype=bool)
    for p in np.where(y > 0)[0]:
        near[max(0, p - NEAR_RADIUS - WINDOW):min(n, p + NEAR_RADIUS + 1)] = True

    
    start = 0
    while start + WINDOW <= n:
        win_end = start + WINDOW - 1
        lbl = window_label(y[start:start + WINDOW])
        if lbl == 0 and near[win_end]:
            count += 1
        start += STRIDE
    return count


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INDEX_CSV)
    df = df.loc[~df["exclude"].apply(to_bool)].reset_index(drop=True)

    # ---- Pass 1: count windows to pre-allocate disk space ----
    print("Pass 1: counting windows...")
    split_counts = {"train": 0, "val": 0, "test": 0}
    for _, row in df.iterrows():
        x_path = os.path.join(PREPROCESS_DIR, f"{row['file_key']}_X_63.npy")
        y_path = os.path.join(LABEL_DIR, f"{row['file_key']}_y.npy")
        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            continue
        X = np.load(x_path)
        y = np.load(y_path)
        if X.shape[0] != y.shape[0]:
            continue
        split_counts[row["split"]] += count_windows_for_file(X, y)
    print("Window counts:", split_counts)

    collected_X = {"train": [], "val": [], "test": []}
    collected_y = {"train": [], "val": [], "test": []}

    # ---- Pass 2: generate and write windows directly to disk ----
    print("Pass 2: writing windows...")
    write_idx = {"train": 0, "val": 0, "test": 0}
    processed, missing, mismatch = 0, 0, 0

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

        collected_X[split].append(Xw)
        collected_y[split].append(yw)
        processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed} files...")

    # ---- Save cleanly with np.save ----
    print("Saving arrays...")
    for split in ("train", "val", "test"):
        if len(collected_X[split]) == 0:
            print(f"WARNING: no windows collected for {split}")
            continue

        X_all = np.concatenate(collected_X[split], axis=0)
        y_all = np.concatenate(collected_y[split], axis=0)

        np.save(os.path.join(OUT_DIR, f"{split}_X.npy"), X_all)
        np.save(os.path.join(OUT_DIR, f"{split}_y.npy"), y_all)

        print(f"\n{split.upper()}:")
        print("  windows:", len(X_all))
        print("  positives:", int((y_all > 0).sum()))
        print("  positive fraction:", float((y_all > 0).mean()) if len(y_all) else 0.0)

    print("\nDone.")
    print("Processed files:", processed)
    print("Missing X or y:", missing)
    print("Mismatched length:", mismatch)
    print("Saved windows to:", OUT_DIR)


if __name__ == "__main__":
    main()
