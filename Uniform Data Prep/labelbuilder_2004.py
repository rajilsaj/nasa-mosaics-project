import os
import numpy as np
import pandas as pd
import yaml

INDEX_CSV = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\dataset_index\dataset_index_2004.csv"
LABEL_DIR = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\zendo-3946033\crossings\labels"
OUT_DIR = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\processed\labels_2004"

WINDOW_MINUTES = 4

os.makedirs(OUT_DIR, exist_ok=True)

def load_change_points(yaml_path: str) -> list[str]:
    with open(yaml_path, "r") as f:
        yml = yaml.safe_load(f)
    return yml.get("change_points", []) or []

def dense_from_change_points(t_ns: np.ndarray, change_points: list[str]) -> np.ndarray:
    """Return dense 0/1 labels aligned to t_ns based on ±WINDOW_MINUTES windows."""
    y = np.zeros(len(t_ns), dtype=np.uint8)
    if not change_points:
        return y

    cp = pd.to_datetime(change_points, format="%d-%m-%Y/%H:%M:%S", errors="coerce", utc=True).dropna()
    if len(cp) == 0:
        return y

    half = pd.Timedelta(minutes=WINDOW_MINUTES).value
    for ts in cp:
        ts_ns = ts.value
        left = np.searchsorted(t_ns, ts_ns - half, side="left")
        right = np.searchsorted(t_ns, ts_ns + half, side="right")
        y[left:right] = 1
    return y

df = pd.read_csv(INDEX_CSV)
df = df[df["exclude"] == False].reset_index(drop=True)

stats = {"files": 0, "bs_files": 0, "mp_files": 0, "both_files": 0, "pos_total": 0, "len_total": 0}

for _, row in df.iterrows():
    file_key = row["file_key"]
    raw_path = row["file_path"]
    has_bs = bool(row["has_bs"])
    has_mp = bool(row["has_mp"])

    raw = pd.read_csv(raw_path)

    # CSV UTC parsing (YEAR-DOY)
    t = pd.to_datetime(raw["UTC"], format="%Y-%jT%H:%M:%S.%f", errors="coerce", utc=True)
    t = t.dropna()
    t_ns = t.astype("int64").to_numpy()

    y_bs = np.zeros(len(t_ns), dtype=np.uint8)
    y_mp = np.zeros(len(t_ns), dtype=np.uint8)

    if has_bs:
        bs_yaml = os.path.join(LABEL_DIR, "bs", "all", f"{file_key}.yaml")
        if os.path.exists(bs_yaml):
            y_bs = dense_from_change_points(t_ns, load_change_points(bs_yaml))

    if has_mp:
        mp_yaml = os.path.join(LABEL_DIR, "mp", "all", f"{file_key}.yaml")
        if os.path.exists(mp_yaml):
            y_mp = dense_from_change_points(t_ns, load_change_points(mp_yaml))

    # Binary label: crossing = BS or MP
    y = ((y_bs == 1) | (y_mp == 1)).astype(np.uint8)

    # Save label array
    out_path = os.path.join(OUT_DIR, f"{file_key}_y.npy")
    np.save(out_path, y)

    # Stats
    stats["files"] += 1
    stats["len_total"] += len(y)
    stats["pos_total"] += int(y.sum())
    if has_bs: stats["bs_files"] += 1
    if has_mp: stats["mp_files"] += 1
    if has_bs and has_mp: stats["both_files"] += 1

print("Done.")
print("Files processed:", stats["files"])
print("BS files:", stats["bs_files"], "MP files:", stats["mp_files"], "Both:", stats["both_files"])
print("Total timesteps:", stats["len_total"])
print("Total positives:", stats["pos_total"])
print("Overall positive fraction:", stats["pos_total"] / max(1, stats["len_total"]))
print("Saved labels to:", OUT_DIR)