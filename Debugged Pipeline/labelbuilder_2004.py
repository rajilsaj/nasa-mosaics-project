import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ===== PATHS =====
INDEX_CSV = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\dataset_index_2004.csv"
LABEL_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\data\zenodo-3946033\crossings\labels"
PREPROCESS_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\preprocess"
OUT_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\labels_2004_2"
# =================

WINDOW_MINUTES = 4
LABEL_MODE = "binary"  # "binary" for TCN, "multiclass" for roadmap/CNN-LSTM
DRY_RUN = False


def to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def find_yaml(crossing_type: str, file_key: str) -> str | None:
    base_dir = os.path.join(LABEL_DIR, crossing_type, "all")
    for ext in (".yaml", ".yml"):
        path = os.path.join(base_dir, f"{file_key}{ext}")
        if os.path.exists(path):
            return path
    return None


def load_change_points(yaml_path: str) -> pd.DatetimeIndex:
    with open(yaml_path, "r") as f:
        yml = yaml.safe_load(f) or {}

    cps = yml.get("change_points", []) or []
    cp = pd.to_datetime(cps, format="%d-%m-%Y/%H:%M:%S", errors="coerce", utc=True)
    if cp.isna().all() and len(cps) > 0:
        cp = pd.to_datetime(cps, errors="coerce", utc=True, dayfirst=True)
    return pd.DatetimeIndex(cp.dropna())

def apply_windows(label_arr: np.ndarray, t: pd.DatetimeIndex, cp: pd.DatetimeIndex, value: int) -> int:
    if len(t) == 0 or len(cp) == 0:
        return 0
    hits = 0
    half = pd.Timedelta(minutes=WINDOW_MINUTES)

    for ts in cp:
        mask = (t >= ts - half) & (t <= ts + half)
        label_arr[np.asarray(mask)] = value
        hits += 1

    return hits


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INDEX_CSV)
    df = df.loc[~df["exclude"].apply(to_bool)].reset_index(drop=True)

    stats = {
        "files": 0,
        "bs_files": 0,
        "mp_files": 0,
        "both_files": 0,
        "pos_total": 0,
        "len_total": 0,
        "bs_cp_total": 0,
        "mp_cp_total": 0,
        "bs_hits": 0,
        "mp_hits": 0,
        "missing_t_ns": 0,
        "missing_bs_yaml": 0,
        "missing_mp_yaml": 0,
    }

    for _, row in df.iterrows():
        file_key = str(row["file_key"]).strip()
        has_bs = to_bool(row["has_bs"])
        has_mp = to_bool(row["has_mp"])

        t_path = os.path.join(PREPROCESS_DIR, f"{file_key}_t_ns.npy")
        if not os.path.exists(t_path):
            stats["missing_t_ns"] += 1
            continue

        t_ns = np.load(t_path)
        t = pd.to_datetime(t_ns, unit='ns', utc=True)

        if LABEL_MODE == "multiclass":
            y = np.zeros(len(t), dtype=np.uint8)
        else:
            y = np.zeros(len(t), dtype=np.uint8)

        if has_bs:
            stats["bs_files"] += 1
            bs_yaml = find_yaml("bs", file_key)
            if bs_yaml is None:
                stats["missing_bs_yaml"] += 1
            else:
                cp_bs = load_change_points(bs_yaml)
                stats["bs_cp_total"] += len(cp_bs)
                bs_value = 1
                stats["bs_hits"] += apply_windows(y, t, cp_bs, bs_value)

        if has_mp:
            stats["mp_files"] += 1
            mp_yaml = find_yaml("mp", file_key)
            if mp_yaml is None:
                stats["missing_mp_yaml"] += 1
            else:
                cp_mp = load_change_points(mp_yaml)
                stats["mp_cp_total"] += len(cp_mp)
                mp_value = 2 if LABEL_MODE == "multiclass" else 1
                stats["mp_hits"] += apply_windows(y, t, cp_mp, mp_value)

        if not DRY_RUN:
            np.save(os.path.join(OUT_DIR, f"{file_key}_y.npy"), y)

        stats["files"] += 1
        stats["len_total"] += len(y)
        stats["pos_total"] += int((y > 0).sum())
        if has_bs and has_mp:
            stats["both_files"] += 1

    print("Done.")
    print("Label mode:", LABEL_MODE)
    print("Dry run:", DRY_RUN)
    print("Files processed:", stats["files"])
    print("BS files:", stats["bs_files"], "MP files:", stats["mp_files"], "Both:", stats["both_files"])
    print("Total timesteps:", stats["len_total"])
    print("Total positives:", stats["pos_total"])
    print("Overall positive fraction:", stats["pos_total"] / max(1, stats["len_total"]))
    print("BS change points parsed:", stats["bs_cp_total"], "hits:", stats["bs_hits"])
    print("MP change points parsed:", stats["mp_cp_total"], "hits:", stats["mp_hits"])
    print("Missing t_ns files:", stats["missing_t_ns"])
    print("Missing BS YAML:", stats["missing_bs_yaml"])
    print("Missing MP YAML:", stats["missing_mp_yaml"])
    if DRY_RUN:
        print("No label files saved.")
    else:
        print("Saved labels to:", OUT_DIR)


if __name__ == "__main__":
    main()
