import os

import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"C:\Users\roger\OneDrive\Desktop\Test Environment\dataset_index\dataset_index_2004.csv"
OUT_DIR = r"C:\Users\roger\OneDrive\Desktop\Test Environment\dataset_index\preprocess"
# =================

ANODE_REDUCE = "sum"  # "sum" or "mean"
NAN_STRATEGY = "zero"  # "zero" or "interp"
LOG_EPS = 1e-6


def find_data_cols(cols: pd.Index) -> list[str]:
    return [c for c in cols if c.startswith("GROUP_1, DATA_")]


def parse_utc(series: pd.Series) -> pd.DatetimeIndex:
    t = pd.to_datetime(series, format="%Y-%jT%H:%M:%S.%f", errors="coerce", utc=True)
    if t.isna().all():
        t = pd.to_datetime(series, errors="coerce", utc=True)
    return pd.DatetimeIndex(t)


def handle_nans(X: np.ndarray) -> np.ndarray:
    if NAN_STRATEGY == "zero":
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if NAN_STRATEGY == "interp":
        Xdf = pd.DataFrame(X)
        Xdf = Xdf.interpolate(method="nearest", axis=0, limit_direction="both")
        return np.nan_to_num(Xdf.to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError("NAN_STRATEGY must be 'zero' or 'interp'")


def preprocess_one_file(raw_path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = pd.read_csv(raw_path)
    t = parse_utc(raw["UTC"])
    valid = ~t.isna()

    raw = raw.loc[valid].copy().reset_index(drop=True)
    t = t[valid]

    raw["parsed_utc"] = t
    raw = raw.sort_values("parsed_utc").reset_index(drop=True)
    t = pd.DatetimeIndex(raw["parsed_utc"])

    data_cols = find_data_cols(raw.columns)
    if len(data_cols) == 0:
        raise RuntimeError(f"No GROUP_1, DATA_* columns found in {raw_path}")
    if len(data_cols) != 504:
        raise RuntimeError(f"Expected 504 data columns, found {len(data_cols)} in {raw_path}")

    X504 = raw[data_cols].to_numpy(dtype=np.float32)
    X = X504.reshape(X504.shape[0], 63, 8)

    if ANODE_REDUCE == "sum":
        X = X.sum(axis=2)
    elif ANODE_REDUCE == "mean":
        X = X.mean(axis=2)
    else:
        raise ValueError("ANODE_REDUCE must be 'sum' or 'mean'")

    X = handle_nans(X)
    X = np.clip(X, 0.0, None)
    X = np.log10(np.clip(X, LOG_EPS, None)).astype(np.float32)

    t_ns = t.astype("int64").to_numpy()
    return t_ns, X


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INDEX_CSV)
    df = df.loc[~df["exclude"].astype(bool)].reset_index(drop=True)

    processed = 0
    for _, row in df.iterrows():
        file_key = row["file_key"]
        raw_path = row["file_path"]

        t_ns, X = preprocess_one_file(raw_path)
        np.save(os.path.join(OUT_DIR, f"{file_key}_t_ns.npy"), t_ns)
        np.save(os.path.join(OUT_DIR, f"{file_key}_X_63.npy"), X)

        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} files...")

    print("Done.")
    print("Saved preprocess arrays to:", OUT_DIR)
    print("Files processed:", processed)


if __name__ == "__main__":
    main()
