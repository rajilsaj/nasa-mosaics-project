#!/usr/bin/env python3
"""
Sanity checks for validation/test sliding window splits and threshold provenance.

This script verifies:
 1. Within-split chronological order and duplicate detection.
 2. Non-overlap between validation and test time ranges.
 3. Consistency of the default decision threshold with validation-derived tuning.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SLIDING_COLS = ["window_id", "start_idx", "end_idx", "start_sclk", "end_sclk", "label"]
DEFAULT_THRESHOLD = 0.45


@dataclass
class SplitCheckResult:
    name: str
    n_rows: int
    duplicate_window_ids: int
    duplicate_rows: int
    chronological_violations: int
    start_sclk_range: Tuple[int, int]
    end_sclk_range: Tuple[int, int]


def load_split(path: Path, name: str) -> SplitCheckResult:
    """Load a sliding-window CSV while avoiding the heavy window payload column."""
    df = pd.read_csv(path, usecols=SLIDING_COLS)

    dup_ids = df.duplicated(subset="window_id").sum()
    dup_rows = df.duplicated().sum()

    start_diff = df["start_sclk"].diff().fillna(0)
    end_diff = df["end_sclk"].diff().fillna(0)
    chronology_violations = int((start_diff < 0).sum() + (end_diff < 0).sum())

    start_range = (int(df["start_sclk"].min()), int(df["start_sclk"].max()))
    end_range = (int(df["end_sclk"].min()), int(df["end_sclk"].max()))

    return SplitCheckResult(
        name=name,
        n_rows=len(df),
        duplicate_window_ids=int(dup_ids),
        duplicate_rows=int(dup_rows),
        chronological_violations=chronology_violations,
        start_sclk_range=start_range,
        end_sclk_range=end_range,
    )


def compare_ranges(val_result: SplitCheckResult, test_result: SplitCheckResult) -> Dict[str, object]:
    """Assess non-overlap between validation and test splits."""
    val_last_end = val_result.end_sclk_range[1]
    test_first_start = test_result.start_sclk_range[0]

    val_idx = pd.read_csv("val_sliding_windows_step10.csv", usecols=["start_idx", "end_idx"])
    test_idx = pd.read_csv("test_sliding_windows_step10.csv", usecols=["start_idx", "end_idx"])

    return {
        "val_end_sclk": val_last_end,
        "test_start_sclk": test_first_start,
        "val_before_test": val_last_end < test_first_start,
        "val_end_idx_max": int(val_idx["end_idx"].max()),
        "test_start_idx_min": int(test_idx["start_idx"].min()),
        "idx_overlap_flag": int(val_idx["end_idx"].max() >= test_idx["start_idx"].min()),
        "note": (
            "Sliding window indices reset per split; overlap flag reflects indexing scheme, "
            "not temporal leakage."
        ),
    }


def find_latest_validation_results(results_dir: Path) -> Optional[Path]:
    """Return the latest validation analysis JSON path, if any."""
    candidates = sorted(results_dir.glob("validation_analysis_results_*.json"))
    return candidates[-1] if candidates else None


def infer_validation_threshold(validation_path: Path) -> Optional[Dict[str, float]]:
    """Determine the validation-derived threshold with the highest F1 score."""
    with open(validation_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results", [])
    if not results:
        return None

    best = max(results, key=lambda r: r.get("f1_score", float("-inf")))
    return {
        "threshold": float(best["threshold"]),
        "f1": float(best["f1_score"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
    }


def run_checks(results_dir: Path, verbose: bool = False) -> None:
    val_path = Path("val_sliding_windows_step10.csv")
    test_path = Path("test_sliding_windows_step10.csv")

    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("Sliding window CSVs not found in the current directory.")

    val_result = load_split(val_path, "validation")
    test_result = load_split(test_path, "test")

    print("=" * 72)
    print("SLIDING WINDOW SPLIT INTEGRITY CHECK")
    print("=" * 72)
    for res in (val_result, test_result):
        print(f"{res.name.capitalize()} split:")
        print(f"  Rows: {res.n_rows:,}")
        print(f"  Window ID duplicates: {res.duplicate_window_ids}")
        print(f"  Full-row duplicates: {res.duplicate_rows}")
        print(f"  Chronology violations (start/end SCLK reversals): {res.chronological_violations}")
        print(f"  start_sclk range: {res.start_sclk_range[0]} -> {res.start_sclk_range[1]}")
        print(f"  end_sclk range:   {res.end_sclk_range[0]} -> {res.end_sclk_range[1]}")
        print("-" * 72)

    range_flags = compare_ranges(val_result, test_result)
    print("Cross-split chronology:")
    print(f"  Validation end_sclk: {range_flags['val_end_sclk']}")
    print(f"  Test start_sclk:     {range_flags['test_start_sclk']}")
    print(f"  val_before_test flag: {'OK' if range_flags['val_before_test'] else 'CHECK'}")
    print(f"  Validation max end_idx: {range_flags['val_end_idx_max']}")
    print(f"  Test min start_idx:     {range_flags['test_start_idx_min']}")
    if range_flags["idx_overlap_flag"]:
        print(f"  Index overlap flag: CHECK (see note: {range_flags['note']})")
    else:
        print("  Index overlap flag: OK")

    latest_validation = find_latest_validation_results(results_dir)
    if latest_validation:
        best = infer_validation_threshold(latest_validation)
        if best:
            print("\nValidation-derived threshold (highest F1):")
            print(f"  File: {latest_validation.name}")
            print(
                f"  Threshold={best['threshold']:.3f} "
                f"(F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f})"
            )
            print(f"  Default test threshold match: {'YES' if best['threshold'] == DEFAULT_THRESHOLD else 'NO'}")
        else:
            print("\nValidation threshold file found, but no results list present.")
    else:
        print("\nNo validation analysis JSON files found to confirm threshold provenance.")

    if verbose:
        # Additional overlap diagnostics if requested
        val_labels = pd.read_csv(val_path, usecols=["label"])
        test_labels = pd.read_csv(test_path, usecols=["label"])
        print("\nLabel distribution check:")
        for name, series in (("Validation", val_labels["label"]), ("Test", test_labels["label"])):
            freq = series.value_counts()
            print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in freq.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate integrity of sliding window splits.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing validation analysis results JSON files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional statistics (label distributions).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_checks(args.results_dir, verbose=args.verbose)

