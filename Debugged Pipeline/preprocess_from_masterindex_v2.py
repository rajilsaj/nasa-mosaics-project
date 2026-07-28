# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd

# ===== PATHS =====
INDEX_CSV = r"/home/jhuss/nasa-mosaics-project/data/dataset_index/dataset_index_2004-2012.csv"
OUT_DIR   = r"/home/jhuss/nasa-mosaics-project/data/dataset_index/preprocess"
# =================

ANODE_REDUCE = "sum"   # "sum" or "mean"
NAN_STRATEGY = "zero"  # "zero" or "interp"
LOG_EPS      = 1e-6

# DT thresholds
DT_DATASET_B_MAX = 2.0   # keep rows where DT <= this for Dataset B
DT_DATASET_A_VAL = 4.0   # target DT value for Dataset A normalisation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess CAPS ELS raw CSVs into X_63 + t_ns arrays."
    )
    parser.add_argument(
        "--dataset",
        choices=["A", "B"],
        required=True,
        help=(
            "Dataset mode: "
            "A = downsample all records to DT=4 (maximum coverage, uniform signal); "
            "B = keep only records where DT <= 2 (high-rate, bowshock-aligned)."
        ),
    )
    parser.add_argument(
        "--filter-dead-time",
        dest="filter_dead_time",
        action="store_true",
        default=False,
        help=(
            "If set, only keep records where DEAD_TIME_METHOD == 2 "
            "(in-flight corrected). Records with value 255 (unknown) are "
            "always dropped regardless of this flag."
        ),
    )
    return parser.parse_args()


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


# ---------------------------------------------------------------------------
# DT filtering / normalisation
# ---------------------------------------------------------------------------

def apply_dt_filter(raw: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """
    Filter or normalise rows based on the DT column and the chosen dataset mode.

    Dataset A: Aggregate all rows with DT < 4 up to DT=4 using a time-weighted
               mean of counts/s (correct because calibrated data is in counts/s).
               Rows with DT == 4 pass through unchanged.
               Rows with DT > 4 are discarded.
               Contiguity check ensures rows being aggregated form a clean 4s window.

    Dataset B: Keep only rows where DT <= DT_DATASET_B_MAX (2.0 s).
    """
    if "DT" not in raw.columns:
        print("  WARNING: DT column not found - skipping DT filter.")
        return raw

    before = len(raw)

    if dataset == "B":
        raw = raw.loc[raw["DT"] <= DT_DATASET_B_MAX].copy()
        after = len(raw)
        dropped = before - after
        if dropped:
            print(f"  DT filter (Dataset B): dropped {dropped}/{before} rows "
                  f"({dropped/before*100:.1f}%), kept {after}")
        return raw

    # --- Dataset A: aggregate up to DT=4 ---
    raw = raw.loc[raw["DT"] <= DT_DATASET_A_VAL].copy().reset_index(drop=True)
    discarded = before - len(raw)
    if discarded:
        print(f"  DT filter (Dataset A): discarded {discarded}/{before} rows with DT > 4")

    data_cols = [c for c in raw.columns if c.startswith("GROUP_1, DATA_")]
    gap_tolerance_s = 0.5  # allow up to 0.5s gap between consecutive records

    out_rows = []
    i = 0
    n = len(raw)
    skipped_incomplete = 0

    while i < n:
        row = raw.iloc[i]
        dt_i = float(row["DT"])

        if dt_i == DT_DATASET_A_VAL:
            out_rows.append(row.to_dict())
            i += 1
            continue

        # Collect rows until we accumulate 4 seconds
        group_indices = [i]
        group_dt = dt_i
        j = i + 1
        gap_detected = False

        while j < n and group_dt < DT_DATASET_A_VAL:
            next_row = raw.iloc[j]
            dt_j = float(next_row["DT"])

            # Contiguity check: actual gap between records is 2*DT (proven empirically —
            # ELS slot width is 2s regardless of accumulation time, so gap = 2 x DT).
            t_current      = raw["parsed_utc"].iloc[group_indices[-1]]
            t_next         = raw["parsed_utc"].iloc[j]
            expected_gap_s = 2.0 * float(raw["DT"].iloc[group_indices[-1]])
            actual_gap_s   = (t_next - t_current).total_seconds()

            if abs(actual_gap_s - expected_gap_s) > gap_tolerance_s:
                gap_detected = True
                break

            group_indices.append(j)
            group_dt += dt_j
            j += 1

        # Only emit if we accumulated exactly 4 seconds and no gap was found
        if not gap_detected and abs(group_dt - DT_DATASET_A_VAL) < 0.01:
            group_df = raw.iloc[group_indices]
            weights = group_df["DT"].to_numpy(dtype=np.float32)
            weights /= weights.sum()

            data_vals = group_df[data_cols].to_numpy(dtype=np.float32)
            aggregated_data = (data_vals * weights[:, None]).sum(axis=0)

            agg_row = raw.iloc[group_indices[0]].to_dict()
            for col, val in zip(data_cols, aggregated_data):
                agg_row[col] = val
            agg_row["DT"] = DT_DATASET_A_VAL
            out_rows.append(agg_row)
        else:
            skipped_incomplete += 1
            # Advance by just one row so we don't skip the rest of the group
            j = i + 1

        i = j

    if skipped_incomplete:
        print(f"  DT aggregation: skipped {skipped_incomplete} incomplete/non-contiguous groups")

    if not out_rows:
        return pd.DataFrame(columns=raw.columns)

    result = pd.DataFrame(out_rows).reset_index(drop=True)
    print(f"  DT filter (Dataset A): {before} rows ? {len(result)} aggregated 4s records")
    return result


def apply_dead_time_filter(raw: pd.DataFrame, filter_dead_time: bool) -> pd.DataFrame:
    """
    Always drop DEAD_TIME_METHOD == 255 (unknown / invalid).
    If filter_dead_time is True, additionally keep only == 2
    (in-flight corrected, the gold-standard for ELS data).
    """
    if "DEAD_TIME_METHOD" not in raw.columns:
        print("  WARNING: DEAD_TIME_METHOD column not found - skipping dead-time filter.")
        return raw

    before = len(raw)

    # Always exclude 255
    raw = raw.loc[raw["DEAD_TIME_METHOD"] != 255].copy()

    if filter_dead_time:
        raw = raw.loc[raw["DEAD_TIME_METHOD"] == 2].copy()

    after = len(raw)
    dropped = before - after
    if dropped:
        mode_label = "== 2 only" if filter_dead_time else "!= 255"
        print(f"  Dead-time filter ({mode_label}): dropped {dropped}/{before} rows "
              f"({dropped/before*100:.1f}%), kept {after}")

    return raw


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------

def preprocess_one_file(
    raw_path: str,
    dataset: str,
    filter_dead_time: bool,
) -> tuple[np.ndarray, np.ndarray]:

    raw = pd.read_csv(raw_path)

    # --- UTC parsing & sort ---
    t = parse_utc(raw["UTC"])
    valid = ~t.isna()
    raw = raw.loc[valid].copy().reset_index(drop=True)
    t = t[valid]
    raw["parsed_utc"] = t
    raw = raw.sort_values("parsed_utc").reset_index(drop=True)
    t = pd.DatetimeIndex(raw["parsed_utc"])

    # --- Quality filters (before extracting X) ---
    raw = apply_dead_time_filter(raw, filter_dead_time)
    raw = apply_dt_filter(raw, dataset)

    # Re-align timestamp index after filtering
    t = pd.DatetimeIndex(raw["parsed_utc"])

    if len(raw) == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 63), dtype=np.float32)

    # --- Extract 504 DATA columns -> (N, 63, 8) -> (N, 63) ---
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Suffix output directory by dataset mode so A and B don't overwrite each other
    out_dir = OUT_DIR.rstrip("/\\") + f"_dataset{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset mode   : {args.dataset}")
    print(f"Filter dead-time to == 2: {args.filter_dead_time}")
    print(f"Output dir     : {out_dir}")
    print()

    df = pd.read_csv(INDEX_CSV)
    df = df.loc[~df["exclude"].astype(bool)].reset_index(drop=True)

    processed = 0
    skipped_empty = 0

    for _, row in df.iterrows():
        file_key = row["file_key"]
        raw_path = row["file_path"]

        t_ns, X = preprocess_one_file(raw_path, args.dataset, args.filter_dead_time)

        if len(t_ns) == 0:
            skipped_empty += 1
            print(f"  SKIP (all rows filtered): {file_key}")
            continue

        np.save(os.path.join(out_dir, f"{file_key}_t_ns.npy"), t_ns)
        np.save(os.path.join(out_dir, f"{file_key}_X_63.npy"), X)

        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} files...")

    print("\nDone.")
    print(f"Output dir     : {out_dir}")
    print(f"Files processed: {processed}")
    print(f"Skipped (empty after filter): {skipped_empty}")


if __name__ == "__main__":
    main()