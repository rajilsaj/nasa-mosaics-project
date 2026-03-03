#!/usr/bin/env python3
"""
Create:
1) Threshold tradeoff plot on validation sliding metrics.
2) Test operating-point bar chart (Baseline@thr vs AE@thr).
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot threshold tradeoff + operating point bars")
    parser.add_argument(
        "--sweep_csv",
        default="results/val_test_threshold_sweep_sliding.csv",
        help="Threshold sweep CSV containing val_/test_ metrics.",
    )
    parser.add_argument(
        "--ae_report_json",
        default="",
        help="AE final test report JSON. If empty, latest is auto-discovered.",
    )
    parser.add_argument(
        "--out_dir",
        default="results/plots_stakeholder_meeting",
        help="Output directory for plots.",
    )
    parser.add_argument("--baseline_threshold", type=float, default=0.74)
    parser.add_argument("--ae_threshold", type=float, default=0.87)
    return parser.parse_args()


def latest(pattern):
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else ""


def load_ae_report(path_hint):
    path = path_hint or latest("results/ae_final_test_report_*.json")
    if not path or not os.path.exists(path):
        raise FileNotFoundError("AE final test report not found.")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return path, payload


def pick_baseline_row(sweep_df, threshold):
    if "threshold" not in sweep_df.columns:
        raise ValueError("Sweep CSV missing 'threshold' column.")
    # nearest threshold to requested value
    idx = (sweep_df["threshold"] - threshold).abs().idxmin()
    return sweep_df.loc[idx]


def make_threshold_tradeoff_plot(df, baseline_thr, ae_thr, out_png):
    required = ["threshold", "val_precision", "val_recall", "val_f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sweep CSV missing columns: {missing}")

    plot_df = df.sort_values("threshold").copy()
    x = plot_df["threshold"].values
    p = plot_df["val_precision"].values
    r = plot_df["val_recall"].values
    f1 = plot_df["val_f1"].values

    plt.figure(figsize=(9, 6))
    plt.plot(x, p, label="Validation Precision", linewidth=2)
    plt.plot(x, r, label="Validation Recall", linewidth=2)
    plt.plot(x, f1, label="Validation F1", linewidth=2)
    plt.axvline(baseline_thr, linestyle="--", linewidth=1.5, label=f"Baseline threshold={baseline_thr:.2f}")
    plt.axvline(ae_thr, linestyle="--", linewidth=1.5, label=f"AE threshold={ae_thr:.2f}")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric")
    plt.title("Threshold Tradeoff (Validation Sliding)")
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def make_operating_point_plot(baseline_metrics, ae_metrics, out_png):
    # panel 1: P/R/F1, panel 2: FP/FN
    metric_names = ["precision", "recall", "f1_score"]
    count_names = ["fp", "fn"]

    base_vals = [baseline_metrics[m] for m in metric_names]
    ae_vals = [ae_metrics[m] for m in metric_names]
    base_counts = [baseline_metrics[c] for c in count_names]
    ae_counts = [ae_metrics[c] for c in count_names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    x1 = np.arange(len(metric_names))
    width = 0.36
    axes[0].bar(x1 - width / 2, base_vals, width=width, label="Baseline RF")
    axes[0].bar(x1 + width / 2, ae_vals, width=width, label="AE-Gated RF")
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(["Precision", "Recall", "F1"])
    axes[0].set_ylim(0, max(max(base_vals), max(ae_vals)) * 1.25 if max(max(base_vals), max(ae_vals)) > 0 else 1)
    axes[0].set_title("Test Sliding Metrics")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    x2 = np.arange(len(count_names))
    axes[1].bar(x2 - width / 2, base_counts, width=width, label="Baseline RF")
    axes[1].bar(x2 + width / 2, ae_counts, width=width, label="AE-Gated RF")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["False Positives", "False Negatives"])
    axes[1].set_title("Test Sliding Error Counts")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    fig.suptitle("Operating Point Comparison (Test Sliding)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sweep_df = pd.read_csv(args.sweep_csv)
    ae_report_path, ae_report = load_ae_report(args.ae_report_json)

    # Baseline metrics come from sweep rows at baseline threshold.
    b = pick_baseline_row(sweep_df, args.baseline_threshold)
    baseline_test_metrics = {
        "precision": float(b["test_precision"]),
        "recall": float(b["test_recall"]),
        "f1_score": float(b["test_f1"]),
        "fp": int(b["test_fp"]),
        "fn": int(b["test_fn"]),
    }

    ae_test = ae_report.get("sliding_test_metrics", {})
    ae_test_metrics = {
        "precision": float(ae_test["precision"]),
        "recall": float(ae_test["recall"]),
        "f1_score": float(ae_test["f1_score"]),
        "fp": int(ae_test["fp"]),
        "fn": int(ae_test["fn"]),
    }

    tradeoff_png = os.path.join(args.out_dir, "threshold_tradeoff_val_sliding.png")
    op_png = os.path.join(args.out_dir, "operating_point_bar_test_sliding.png")
    summary_json = os.path.join(args.out_dir, "operating_point_summary.json")

    make_threshold_tradeoff_plot(sweep_df, args.baseline_threshold, args.ae_threshold, tradeoff_png)
    make_operating_point_plot(baseline_test_metrics, ae_test_metrics, op_png)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep_csv": args.sweep_csv,
                "ae_report_json": ae_report_path,
                "baseline_threshold": args.baseline_threshold,
                "ae_threshold": args.ae_threshold,
                "baseline_test_metrics": baseline_test_metrics,
                "ae_test_metrics": ae_test_metrics,
            },
            f,
            indent=2,
        )

    print("Saved:")
    print(f"  {tradeoff_png}")
    print(f"  {op_png}")
    print(f"  {summary_json}")


if __name__ == "__main__":
    main()

