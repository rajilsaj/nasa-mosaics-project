#!/usr/bin/env python3
"""
Temporal split only for Mars vortex pipeline.

This script only performs chronological train/val/test splitting with gaps.
Window extraction is handled separately by extract_windows.py.
"""

import os
import pandas as pd
import argparse

TRAIN_RATIO = 0.60
GAP_RATIO = 0.005
VAL_RATIO = 0.15

ML_FILE = "ml_ready_vortex_data.csv"
JACKSON_FILE = "Jackson_vortex_detections_reformatted_augmented.csv"
OUTPUT_DIR = "temporal_splits"


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal split only (no extraction)")
    return parser.parse_args()


def main():
    parse_args()
    print("=" * 70)
    print("TEMPORAL SPLITTING ONLY")
    print("=" * 70)

    ml_df = pd.read_csv(ML_FILE)
    jackson_df = pd.read_csv(JACKSON_FILE)

    ml_df["SCLK"] = pd.to_numeric(ml_df["SCLK"], errors="coerce")
    jackson_df["SCLK"] = pd.to_numeric(jackson_df["SCLK"], errors="coerce")
    ml_df = ml_df.dropna(subset=["SCLK"]).sort_values("SCLK").reset_index(drop=True)
    jackson_df = jackson_df.dropna(subset=["SCLK"]).sort_values("SCLK").reset_index(drop=True)

    n = len(ml_df)
    train_end_idx = int(n * TRAIN_RATIO)
    val_start_idx = int(n * (TRAIN_RATIO + GAP_RATIO))
    val_end_idx = int(n * (TRAIN_RATIO + GAP_RATIO + VAL_RATIO))
    test_start_idx = int(n * (TRAIN_RATIO + 2 * GAP_RATIO + VAL_RATIO))

    ml_train = ml_df.iloc[:train_end_idx].copy()
    ml_val = ml_df.iloc[val_start_idx:val_end_idx].copy()
    ml_test = ml_df.iloc[test_start_idx:].copy()

    train_sclk_max = ml_train["SCLK"].max()
    val_sclk_min = ml_val["SCLK"].min()
    val_sclk_max = ml_val["SCLK"].max()
    test_sclk_min = ml_test["SCLK"].min()

    jackson_train = jackson_df[jackson_df["SCLK"] <= train_sclk_max].copy()
    jackson_val = jackson_df[
        (jackson_df["SCLK"] >= val_sclk_min) & (jackson_df["SCLK"] <= val_sclk_max)
    ].copy()
    jackson_test = jackson_df[jackson_df["SCLK"] >= test_sclk_min].copy()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ml_train.to_csv(os.path.join(OUTPUT_DIR, "ml_train.csv"), index=False)
    ml_val.to_csv(os.path.join(OUTPUT_DIR, "ml_val.csv"), index=False)
    ml_test.to_csv(os.path.join(OUTPUT_DIR, "ml_test.csv"), index=False)
    jackson_train.to_csv(os.path.join(OUTPUT_DIR, "jackson_train.csv"), index=False)
    jackson_val.to_csv(os.path.join(OUTPUT_DIR, "jackson_val.csv"), index=False)
    jackson_test.to_csv(os.path.join(OUTPUT_DIR, "jackson_test.csv"), index=False)

    print(f"Saved splits to {OUTPUT_DIR}/")
    print(f"Train/Val/Test ML sizes: {len(ml_train):,}/{len(ml_val):,}/{len(ml_test):,}")
    print(
        f"Train/Val/Test Jackson events: {len(jackson_train)}/{len(jackson_val)}/{len(jackson_test)}"
    )


if __name__ == "__main__":
    main()

