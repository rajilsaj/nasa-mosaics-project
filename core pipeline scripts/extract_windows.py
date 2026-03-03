#!/usr/bin/env python3
"""
Window extraction only for Mars vortex pipeline.

Reads precomputed temporal splits from temporal_splits/ and extracts
event-local, mission-aligned positive windows ending inside gt_detection_win.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

SPLITS_DIR = "temporal_splits"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract windows from existing splits")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Which split to process (default: all)",
    )
    parser.add_argument("--window_size", type=int, default=60)
    return parser.parse_args()


def extract_windows_from_split(ml_df, jackson_df, split_name, window_size):
    ml_df_reset = ml_df.reset_index(drop=True)
    all_windows = []
    failed = 0

    for _, row in tqdm(
        jackson_df.iterrows(), total=len(jackson_df), desc=f"Processing {split_name} events"
    ):
        target_sclk = row["SCLK"]
        matches = ml_df_reset[ml_df_reset["SCLK"] == target_sclk]
        if matches.empty:
            failed += 1
            continue

        target_pos = matches.index[0]
        search_space = ml_df_reset.iloc[: target_pos + 1]
        precursor = search_space[search_space["gt_detection_win"] == True]
        if precursor.empty:
            failed += 1
            continue

        last_true = precursor.index[-1]
        end_pos = last_true
        start_pos = end_pos - window_size + 1
        if start_pos < 0:
            failed += 1
            continue

        window = ml_df_reset.iloc[start_pos : end_pos + 1].copy()
        if len(window) < window_size:
            failed += 1
            continue

        window["window_id"] = len(all_windows)
        window["event_sclk"] = target_sclk
        window["split"] = split_name
        window["label"] = True
        all_windows.append(window)

    success = len(all_windows)
    print(f"{split_name.upper()} extracted windows: {success} (failed: {failed})")
    if success == 0:
        return None
    result_df = pd.concat(all_windows, ignore_index=True)

    span_stats = (
        result_df.groupby("window_id")
        .agg(start_sclk=("SCLK", "min"), end_sclk=("SCLK", "max"))
        .reset_index()
    )
    dup_spans = int(span_stats.duplicated(subset=["start_sclk", "end_sclk"]).sum())
    print(f"  Unique spans: {len(span_stats) - dup_spans}/{len(span_stats)}")
    print(f"  Duplicate span count: {dup_spans}")
    return result_df


def main():
    args = parse_args()
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
        out_df = extract_windows_from_split(ml_df, jackson_df, split_name, args.window_size)
        if out_df is not None:
            out_file = f"{split_name}_windows.csv"
            out_df.to_csv(out_file, index=False)
            print(f"  Saved: {out_file} ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()

