import os
import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/dataset_index_2005.csv"
PREPROCESS_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/preprocess"
LABEL_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/labels_2005"
OUT_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/windows_2005_focus6h"
# =================

WINDOW = 64
STRIDE = 16
EVENT_HOURS = 6  # keep TRAIN windows whose center is within ±6h of any positive timestep


def get_doy(file_key: str) -> int:
    return int(file_key[8:11])


def get_split(doy: int) -> str:
    if doy <= 270:
        return "train"
    if doy <= 335:
        return "val"
    return "test"


def make_windows_all(X: np.ndarray, y: np.ndarray):
    half = WINDOW // 2
    X_out, y_out = [], []

    for start in range(0, len(X) - WINDOW + 1, STRIDE):
        end = start + WINDOW
        center = start + half
        X_out.append(X[start:end])
        y_out.append(int(y[center]))

    if len(X_out) == 0:
        return (
            np.empty((0, WINDOW, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.uint8)


def make_windows_focus(X: np.ndarray, y: np.ndarray, t_ns: np.ndarray):
    """
    Keep windows whose CENTER timestamp is within ±EVENT_HOURS of any positive timestep.
    This keeps all positives and nearby background, and drops far-away background.
    """
    half = WINDOW // 2
    X_out, y_out = [], []

    pos_times = np.sort(t_ns[y > 0])
    if len(pos_times) == 0:
        return (
            np.empty((0, WINDOW, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    radius_ns = int(EVENT_HOURS * 3600 * 1e9)

    for start in range(0, len(X) - WINDOW + 1, STRIDE):
        end = start + WINDOW
        center = start + half
        center_t = t_ns[center]

        idx = np.searchsorted(pos_times, center_t)

        nearest = None
        if idx < len(pos_times):
            nearest = abs(int(pos_times[idx]) - int(center_t))
        if idx > 0:
            prev_diff = abs(int(pos_times[idx - 1]) - int(center_t))
            nearest = prev_diff if nearest is None else min(nearest, prev_diff)

        if nearest is not None and nearest <= radius_ns:
            X_out.append(X[start:end])
            y_out.append(int(y[center]))

    if len(X_out) == 0:
        return (
            np.empty((0, WINDOW, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
        )

    return np.asarray(X_out, dtype=np.float32), np.asarray(y_out, dtype=np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INDEX_CSV)
    df = df.loc[~df["exclude"].astype(bool)].reset_index(drop=True)

    splits_X = {"train": [], "val": [], "test": []}
    splits_y = {"train": [], "val": [], "test": []}

    processed = 0
    missing = 0
    mismatch = 0

    for _, row in df.iterrows():
        file_key = str(row["base"]).strip()
        split = get_split(get_doy(file_key))

        x_path = os.path.join(PREPROCESS_DIR, f"{file_key}_X_63.npy")
        y_path = os.path.join(LABEL_DIR, f"{file_key}_y.npy")
        t_path = os.path.join(PREPROCESS_DIR, f"{file_key}_t_ns.npy")

        if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(t_path)):
            missing += 1
            continue

        X = np.load(x_path)
        y = np.load(y_path)
        t_ns = np.load(t_path)

        if not (X.shape[0] == y.shape[0] == t_ns.shape[0]):
            mismatch += 1
            print(f"Length mismatch for {file_key}: X={X.shape[0]} y={y.shape[0]} t={t_ns.shape[0]}")
            continue

        # Focus TRAIN only; keep VAL/TEST natural for honest evaluation
        if split == "train":
            Xw, yw = make_windows_focus(X, y, t_ns)
        else:
            Xw, yw = make_windows_all(X, y)

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
    print("Missing X/y/t:", missing)
    print("Mismatched length:", mismatch)
    print("Saved focused windows to:", OUT_DIR)


if __name__ == "__main__":
    main()
