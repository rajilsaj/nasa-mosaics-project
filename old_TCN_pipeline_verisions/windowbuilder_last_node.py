import os
import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\dataset_index_2004.csv"
PREPROCESS_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\preprocess"
LABEL_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\labels_2004_2"
OUT_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\windows_2004"
# =================

WINDOW = 128                            #was 128
STRIDE_FAR = 16
STRIDE_NEAR = 4
NEAR_RADIUS = 75
N_FEATURES = 63

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
        lo = max(0, p - NEAR_RADIUS - WINDOW)
        hi = min(n, p + NEAR_RADIUS + 1)
        near[lo:hi] = True

    start = 0
    while start + WINDOW <= n:
        end = start + WINDOW - 1    # uses last timestamp in the window
        stride = STRIDE_NEAR if near[end] else STRIDE_FAR
        X_out.append(X[start:start + WINDOW])
        y_out.append(int(y[end]))
        start += stride

    if len(X_out) == 0:
        return np.empty((0, WINDOW, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.uint8)

    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.uint8)


def count_windows_for_file(X: np.ndarray, y: np.ndarray) -> int:
    """
    Dry-run of the window loop — counts how many windows a file will produce
    without storing anything. Used in Pass 1 to pre-allocate disk space.
    """
    n = X.shape[0]
    pos_idx = np.where(y > 0)[0]
    near = np.zeros(n, dtype=bool)
    for p in pos_idx:
        near[max(0, p - NEAR_RADIUS - WINDOW):min(n, p + NEAR_RADIUS + 1)] = True

    count = 0
    start = 0
    while start + WINDOW <= n:
        end = start + WINDOW - 1
        stride = STRIDE_NEAR if near[end] else STRIDE_FAR
        count += 1
        start += stride
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

    # ---- Pre-allocate memmaps on disk ----
    mmap_X, mmap_y = {}, {}
    for split, count in split_counts.items():
        mmap_X[split] = np.memmap(
            os.path.join(OUT_DIR, f"{split}_X.npy"),
            dtype=np.float32, mode="w+",
            shape=(count, WINDOW, N_FEATURES),
        )
        mmap_y[split] = np.memmap(
            os.path.join(OUT_DIR, f"{split}_y.npy"),
            dtype=np.uint8, mode="w+",
            shape=(count,),
        )

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

        idx = write_idx[split]
        n = len(Xw)
        mmap_X[split][idx:idx + n] = Xw
        mmap_y[split][idx:idx + n] = yw
        write_idx[split] += n
        processed += 1

        if processed % 100 == 0:
            print(f"Processed {processed} files...")

    # ---- Flush and print stats ----
    for split in ("train", "val", "test"):
        mmap_X[split].flush()
        mmap_y[split].flush()
        y_all = mmap_y[split]
        print(f"\n{split.upper()}:")
        print("  windows:", split_counts[split])
        print("  positives:", int((y_all > 0).sum()))
        print("  positive fraction:", float((y_all > 0).mean()) if split_counts[split] else 0.0)

    print("\nDone.")
    print("Processed files:", processed)
    print("Missing X or y:", missing)
    print("Mismatched length:", mismatch)
    print("Saved windows to:", OUT_DIR)


if __name__ == "__main__":
    main()
