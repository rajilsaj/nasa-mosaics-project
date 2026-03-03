#!/usr/bin/env python3
"""
Window extraction only for comprehensive pipeline.

Reads split files from data/splits and writes windows to data/windows.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(SCRIPT_DIR, "data", "splits")
WINDOWS_DIR = os.path.join(SCRIPT_DIR, "data", "windows")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract windows from comprehensive splits")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--window_size", type=int, default=60)
    return parser.parse_args()


def extract_windows_from_split(ml_df, jackson_df, split_name, window_size):
    ml_df_reset = ml_df.reset_index(drop=True)
    windows = []
    failed = 0

    for _, row in tqdm(
        jackson_df.iterrows(), total=len(jackson_df), desc=f"Processing {split_name} events"
    ):
        target_sclk = row["SCLK"]
        target_matches = ml_df_reset[ml_df_reset["SCLK"] == target_sclk]
        if target_matches.empty:
            failed += 1
            continue

        target_pos = target_matches.index[0]
        search_space = ml_df_reset.iloc[: target_pos + 1]
        precursor = search_space[search_space["gt_detection_win"] == True]
        if precursor.empty:
            failed += 1
            continue

        last_true_pos = precursor.index[-1]
        end_pos = last_true_pos
        start_pos = end_pos - window_size + 1

        if start_pos < 0:
            failed += 1
            continue

        window = ml_df_reset.iloc[start_pos : end_pos + 1].copy()
        if len(window) < window_size:
            failed += 1
            continue

        window["window_id"] = len(windows)
        window["event_sclk"] = target_sclk
        window["split"] = split_name
        window["label"] = True
        windows.append(window)

    print(f"{split_name.upper()} extracted windows: {len(windows)} (failed: {failed})")
    if not windows:
        return None

    out_df = pd.concat(windows, ignore_index=True)
    span_stats = (
        out_df.groupby("window_id")
        .agg(start_sclk=("SCLK", "min"), end_sclk=("SCLK", "max"))
        .reset_index()
    )
    dup_spans = int(span_stats.duplicated(subset=["start_sclk", "end_sclk"]).sum())
    print(f"  Unique spans: {len(span_stats) - dup_spans}/{len(span_stats)}")
    print(f"  Duplicate span count: {dup_spans}")
    return out_df


def main():
    args = parse_args()
    os.makedirs(WINDOWS_DIR, exist_ok=True)
    targets = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split_name in targets:
        ml_path = os.path.join(SPLITS_DIR, f"ml_{split_name}.csv")
        jackson_path = os.path.join(SPLITS_DIR, f"jackson_{split_name}.csv")
        if not os.path.exists(ml_path) or not os.path.exists(jackson_path):
            raise FileNotFoundError(
                f"Missing split files for {split_name}. Run split_data.py first."
            )

        ml_df = pd.read_csv(ml_path)
        jackson_df = pd.read_csv(jackson_path)
        result = extract_windows_from_split(ml_df, jackson_df, split_name, args.window_size)
        if result is not None:
            out_path = os.path.join(WINDOWS_DIR, f"{split_name}_windows.csv")
            result.to_csv(out_path, index=False)
            print(f"  Saved: {out_path} ({len(result):,} rows)")


if __name__ == "__main__":
    main()

