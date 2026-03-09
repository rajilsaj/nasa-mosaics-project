import os
from pathlib import Path

import numpy as np
import pandas as pd

# ========= UPDATE THESE =========
INDEX_CSV = r"C:\Users\roger\OneDrive\Desktop\Cassini New\Spacecraft Data\dataset_index\dataset_index_2004.csv"
OUT_DIR = r"C:\Users\roger\OneDrive\Desktop\Cassini New\Spacecraft Data\dataset_index\preprocess"
# =================================

# Choose how to reduce 8 anodes -> 1 value per energy bin
ANODE_REDUCE = "sum"  # "sum" or "mean"

# NaN handling
NAN_STRATEGY = "zero"  # "zero" or "interp"

def find_data_cols(cols):
    # signal columns are: 'GROUP_1, DATA_1' ... 'GROUP_1, DATA_504'
    return [c for c in cols if c.startswith("GROUP_1, DATA_")]

def parse_utc(series: pd.Series) -> pd.DatetimeIndex:
    # YEAR-DOY format in your raw CSV: 2004-153T01:10:34.762
    return pd.to_datetime(series, format="%Y-%jT%H:%M:%S.%f", errors="coerce", utc=True)

def handle_nans(X: np.ndarray) -> np.ndarray:
    if NAN_STRATEGY == "zero":
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if NAN_STRATEGY == "interp":
        # interpolate along time for each energy bin
        # (simple + safe; if all-NaN column exists, it will remain NaN -> filled with 0)
        Xdf = pd.DataFrame(X)
        Xdf = Xdf.interpolate(method="nearest", axis=0, limit_direction="both")
        X = Xdf.to_numpy()
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError("NAN_STRATEGY must be 'zero' or 'interp'")

def preprocess_one_file(raw_path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(raw_path)

    # 1) Parse UTC into timezone-aware datetime
    t = parse_utc(raw["UTC"])
    valid = t.notna()

    raw = raw.loc[valid].reset_index(drop=True)
    t = t.loc[valid].reset_index(drop=True)

    # 2) Drop metadata columns (we simply don't use them)
    #    UTC, DEAD_TIME_METHOD, TELEMETRY, DT

    # 3) Keep only signal columns
    data_cols = find_data_cols(raw.columns)
    if len(data_cols) == 0:
        raise RuntimeError(f"No GROUP_1, DATA_* columns found in {raw_path}")
    if len(data_cols) != 504:
        # Not fatal, but important to know if a file differs
        print(f"Warning: expected 504 DATA cols, found {len(data_cols)} in {raw_path}")

    X504 = raw[data_cols].to_numpy(dtype=np.float32)  # 4) cast float32

    # 5) Reshape to (time, 63, 8) then reduce to (time, 63)
    # If a file isn't exactly 504 cols, this reshape will fail → that’s good (forces you to notice)
    X = X504.reshape(X504.shape[0], 63, 8)

    if ANODE_REDUCE == "sum":
        X = X.sum(axis=2)
    elif ANODE_REDUCE == "mean":
        X = X.mean(axis=2)
    else:
        raise ValueError("ANODE_REDUCE must be 'sum' or 'mean'")

    # 6) Handle NaNs
    X = handle_nans(X)

    # 7) Handle negative values (clip to 0)
    X = np.clip(X, 0.0, None).astype(np.float32)

    # return UTC times as int64 ns (fast + timezone-safe) + X
    t_ns = t.astype("int64").to_numpy()
    return t_ns, X

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INDEX_CSV)

    # Load only exclude=False
    df = df[df["exclude"] == False].reset_index(drop=True)

    processed = 0
    for _, row in df.iterrows():
        file_key = row["file_key"]
        raw_path = row["file_path"]

        t_ns, X = preprocess_one_file(raw_path)

        # Save per file (fast + simple)
        np.save(os.path.join(OUT_DIR, f"{file_key}_t_ns.npy"), t_ns)
        np.save(os.path.join(OUT_DIR, f"{file_key}_X_63.npy"), X)

        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} files...")

    print("Done.")
    print("Saved to:", OUT_DIR)
    print("Files processed:", processed)

if __name__ == "__main__":
    main()