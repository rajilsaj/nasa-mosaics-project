#!/usr/bin/env python3
"""
Temporal split only for comprehensive pipeline.
"""

import os
import pandas as pd

TRAIN_RATIO = 0.60
GAP_RATIO = 0.005
VAL_RATIO = 0.15

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "splits")


def detect_comprehensive_file():
    candidates = [
        os.path.join(PARENT_DIR, "comprehensive_filtered_data_optimized.csv"),
        os.path.join(PARENT_DIR, "comprehensive_filtered_data_optimized (1).csv"),
        os.path.join(SCRIPT_DIR, "comprehensive_filtered_data_optimized.csv"),
        os.path.join(SCRIPT_DIR, "comprehensive_filtered_data_optimized (1).csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def main():
    print("=" * 70)
    print("COMPREHENSIVE TEMPORAL SPLITTING ONLY")
    print("=" * 70)

    comprehensive_file = detect_comprehensive_file()
    jackson_file = os.path.join(PARENT_DIR, "Jackson_vortex_detections_reformatted_augmented.csv")

    ml_df = pd.read_csv(comprehensive_file)
    jackson_df = pd.read_csv(jackson_file)

    ml_df["SCLK"] = pd.to_numeric(ml_df["SCLK"], errors="coerce")
    jackson_df["SCLK"] = pd.to_numeric(jackson_df["SCLK"], errors="coerce")
    ml_df = ml_df.dropna(subset=["SCLK"]).sort_values("SCLK").reset_index(drop=True)
    jackson_df = jackson_df.dropna(subset=["SCLK"]).sort_values("SCLK").reset_index(drop=True)

    n = len(ml_df)
    train_end = int(n * TRAIN_RATIO)
    val_start = int(n * (TRAIN_RATIO + GAP_RATIO))
    val_end = int(n * (TRAIN_RATIO + GAP_RATIO + VAL_RATIO))
    test_start = int(n * (TRAIN_RATIO + 2 * GAP_RATIO + VAL_RATIO))

    ml_train = ml_df.iloc[:train_end].copy()
    ml_val = ml_df.iloc[val_start:val_end].copy()
    ml_test = ml_df.iloc[test_start:].copy()

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

    print(f"Input file: {comprehensive_file}")
    print(f"Saved splits: {OUTPUT_DIR}")
    print(f"Train/Val/Test ML: {len(ml_train):,}/{len(ml_val):,}/{len(ml_test):,}")
    print(f"Train/Val/Test events: {len(jackson_train)}/{len(jackson_val)}/{len(jackson_test)}")


if __name__ == "__main__":
    main()

