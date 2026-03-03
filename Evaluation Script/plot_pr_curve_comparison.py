#!/usr/bin/env python3
"""
Plot PR-curve comparison for baseline RF vs AE-gated RF on sliding windows.

Outputs:
- results/plots_stakeholder_meeting/pr_curve_val_sliding.png
- results/plots_stakeholder_meeting/pr_curve_test_sliding.png
- results/plots_stakeholder_meeting/pr_curve_points.csv
"""

import argparse
import glob
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


EXCLUDE_SLIDING = {
    "window_id",
    "start_idx",
    "end_idx",
    "start_sclk",
    "end_sclk",
    "event_sclk",
    "split",
    "label",
    "sliding_window_id",
    "sliding_start_idx",
    "sliding_end_idx",
    "sliding_start_sclk",
    "sliding_end_sclk",
}


def parse_args():
    parser = argparse.ArgumentParser(description="PR comparison: baseline RF vs AE-gated RF")
    parser.add_argument("--val_features", default="val_sliding_features.csv")
    parser.add_argument("--test_features", default="test_sliding_features.csv")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results/plots_stakeholder_meeting")
    parser.add_argument("--rf_model_path", default="")
    parser.add_argument("--rf_metadata_path", default="")
    parser.add_argument("--ae_best_config_path", default="")
    parser.add_argument(
        "--rf_operating_threshold",
        type=float,
        default=0.74,
        help="Marker threshold for baseline RF point on curves.",
    )
    parser.add_argument(
        "--ae_operating_threshold",
        type=float,
        default=None,
        help="Marker threshold for AE model. If unset, uses best-config threshold.",
    )
    return parser.parse_args()


def latest(pattern):
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else ""


def pick_paths(args):
    rf_model = args.rf_model_path or latest(os.path.join(args.models_dir, "rf_vortex_detector_*.pkl"))
    rf_meta = args.rf_metadata_path or latest(os.path.join(args.models_dir, "model_metadata_*.json"))
    ae_cfg = args.ae_best_config_path or latest(os.path.join("results", "ae_validation_best_config_*.json"))
    if not rf_model or not os.path.exists(rf_model):
        raise FileNotFoundError("Baseline RF model not found.")
    if not rf_meta or not os.path.exists(rf_meta):
        raise FileNotFoundError("Baseline RF metadata not found.")
    if not ae_cfg or not os.path.exists(ae_cfg):
        raise FileNotFoundError("AE best-config json not found.")
    return rf_model, rf_meta, ae_cfg


def normalize_sliding_labels(df):
    out = df.copy()
    out["label"] = out["label"].astype(str)
    out = out[out["label"] != "Omit"].copy()
    out["label"] = out["label"].map({"True": 1, "False": 0, "1": 1, "0": 0})
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out


def infer_features(df):
    return [c for c in df.columns if c not in EXCLUDE_SLIDING]


def get_operating_point(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def evaluate_model_curves(model, feature_cols, df):
    eval_df = normalize_sliding_labels(df)
    missing = [c for c in feature_cols if c not in eval_df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = eval_df[feature_cols].values
    y = eval_df["label"].values
    y_prob = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    ap = average_precision_score(y, y_prob)
    prevalence = float(y.mean()) if len(y) > 0 else 0.0
    return y, y_prob, precision, recall, float(ap), prevalence


def plot_one(split_name, out_path, baseline_data, ae_data, rf_thr, ae_thr):
    y_b, prob_b, p_b, r_b, ap_b, prev = baseline_data
    y_a, prob_a, p_a, r_a, ap_a, _ = ae_data

    rb_p, rb_r = get_operating_point(y_b, prob_b, rf_thr)
    ae_p, ae_r = get_operating_point(y_a, prob_a, ae_thr)

    plt.figure(figsize=(8, 6))
    plt.plot(r_b, p_b, linewidth=2.0, label=f"Baseline RF (AUPRC={ap_b:.4f})")
    plt.plot(r_a, p_a, linewidth=2.0, label=f"AE-Gated RF (AUPRC={ap_a:.4f})")
    plt.axhline(prev, linestyle="--", linewidth=1.2, label=f"Prevalence={prev:.4f}")
    plt.scatter([rb_r], [rb_p], s=70, marker="o", label=f"RF @ {rf_thr:.2f}")
    plt.scatter([ae_r], [ae_p], s=70, marker="s", label=f"AE @ {ae_thr:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({split_name})")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

    return {
        "split": split_name,
        "prevalence": prev,
        "rf_auprc": ap_b,
        "ae_auprc": ap_a,
        "rf_threshold": rf_thr,
        "ae_threshold": ae_thr,
        "rf_precision_at_threshold": rb_p,
        "rf_recall_at_threshold": rb_r,
        "ae_precision_at_threshold": ae_p,
        "ae_recall_at_threshold": ae_r,
    }


def append_curve_points(rows, split, model_tag, precisions, recalls):
    for p, r in zip(precisions, recalls):
        rows.append(
            {
                "split": split,
                "model": model_tag,
                "precision": float(p),
                "recall": float(r),
            }
        )


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    rf_model_path, rf_meta_path, ae_cfg_path = pick_paths(args)
    with open(rf_meta_path, "r", encoding="utf-8") as f:
        rf_meta = json.load(f)
    with open(ae_cfg_path, "r", encoding="utf-8") as f:
        ae_cfg = json.load(f)

    ae_model_path = ae_cfg["rf_model_path"]
    if not os.path.exists(ae_model_path):
        raise FileNotFoundError(f"AE RF model missing: {ae_model_path}")

    rf_features = rf_meta.get("feature_names", [])
    ae_features = ae_cfg.get("feature_cols", [])
    if not rf_features or not ae_features:
        raise ValueError("Could not resolve feature columns from metadata/best-config.")

    ae_thr = args.ae_operating_threshold
    if ae_thr is None:
        ae_thr = float(ae_cfg.get("threshold", 0.5))

    val_df = pd.read_csv(args.val_features)
    test_df = pd.read_csv(args.test_features)

    # Fallback feature inference if needed.
    if not rf_features:
        rf_features = infer_features(val_df)
    if not ae_features:
        ae_features = infer_features(val_df)

    baseline_model = joblib.load(rf_model_path)
    ae_model = joblib.load(ae_model_path)

    val_baseline = evaluate_model_curves(baseline_model, rf_features, val_df)
    val_ae = evaluate_model_curves(ae_model, ae_features, val_df)
    test_baseline = evaluate_model_curves(baseline_model, rf_features, test_df)
    test_ae = evaluate_model_curves(ae_model, ae_features, test_df)

    val_png = os.path.join(args.results_dir, "pr_curve_val_sliding.png")
    test_png = os.path.join(args.results_dir, "pr_curve_test_sliding.png")

    val_summary = plot_one("Validation Sliding", val_png, val_baseline, val_ae, args.rf_operating_threshold, ae_thr)
    test_summary = plot_one("Test Sliding", test_png, test_baseline, test_ae, args.rf_operating_threshold, ae_thr)

    point_rows = []
    append_curve_points(point_rows, "val", "baseline_rf", val_baseline[2], val_baseline[3])
    append_curve_points(point_rows, "val", "ae_gated_rf", val_ae[2], val_ae[3])
    append_curve_points(point_rows, "test", "baseline_rf", test_baseline[2], test_baseline[3])
    append_curve_points(point_rows, "test", "ae_gated_rf", test_ae[2], test_ae[3])
    points_csv = os.path.join(args.results_dir, "pr_curve_points.csv")
    pd.DataFrame(point_rows).to_csv(points_csv, index=False)

    summary_json = os.path.join(args.results_dir, "pr_curve_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_model_path": rf_model_path,
                "baseline_metadata_path": rf_meta_path,
                "ae_best_config_path": ae_cfg_path,
                "ae_model_path": ae_model_path,
                "rf_operating_threshold": args.rf_operating_threshold,
                "ae_operating_threshold": ae_thr,
                "validation": val_summary,
                "test": test_summary,
            },
            f,
            indent=2,
        )

    print("Saved:")
    print(f"  {val_png}")
    print(f"  {test_png}")
    print(f"  {points_csv}")
    print(f"  {summary_json}")
    print("\nValidation AUPRC:")
    print(f"  Baseline RF: {val_summary['rf_auprc']:.4f}")
    print(f"  AE-Gated RF: {val_summary['ae_auprc']:.4f}")
    print("Test AUPRC:")
    print(f"  Baseline RF: {test_summary['rf_auprc']:.4f}")
    print(f"  AE-Gated RF: {test_summary['ae_auprc']:.4f}")


if __name__ == "__main__":
    main()

