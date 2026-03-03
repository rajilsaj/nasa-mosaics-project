#!/usr/bin/env python3
"""
Sliding Window Evaluation - Pipeline-Aligned Version
====================================================

Evaluates a pre-trained RF model on precomputed sliding-window features.
This script does NOT retrain. It loads:
- model artifact (*.pkl)
- metadata (*.json) to get feature order and selected threshold
- val/test sliding feature CSVs
"""

import argparse
import glob
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

EXCLUDE_COLUMNS = {
    "window_id",
    "start_idx",
    "end_idx",
    "start_sclk",
    "end_sclk",
    "event_sclk",
    "split",
    "label",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RF on sliding features")
    parser.add_argument("--val_features", default="val_sliding_features.csv")
    parser.add_argument("--test_features", default="test_sliding_features.csv")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--metadata_path", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override threshold. If unset, uses metadata selected_threshold.",
    )
    return parser.parse_args()


def latest_file(pattern):
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def resolve_paths(args):
    model_path = args.model_path or latest_file(os.path.join(args.models_dir, "rf_vortex_detector_*.pkl"))
    metadata_path = args.metadata_path or latest_file(os.path.join(args.models_dir, "model_metadata_*.json"))
    if model_path is None:
        raise FileNotFoundError("No model artifact found. Train model first.")
    if metadata_path is None:
        raise FileNotFoundError("No model metadata JSON found. Train model first.")
    return model_path, metadata_path


def normalize_labels(df):
    out = df.copy()
    out["label"] = out["label"].astype(str)
    out = out[out["label"] != "Omit"].copy()
    out["label"] = out["label"].map({"True": 1, "False": 0, "1": 1, "0": 0})
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out


def infer_feature_names(df):
    return [c for c in df.columns if c not in EXCLUDE_COLUMNS]


def evaluate_split(model, df, feature_names, threshold, split_name):
    eval_df = normalize_labels(df)
    missing_cols = [c for c in feature_names if c not in eval_df.columns]
    if missing_cols:
        raise ValueError(f"{split_name}: missing required feature columns: {missing_cols}")

    X = eval_df[feature_names].values
    y = eval_df["label"].values
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else None

    print(f"\n{'=' * 70}")
    print(f"EVALUATION RESULTS ON {split_name.upper()} SLIDING WINDOWS")
    print(f"{'=' * 70}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Windows (after Omit filter): {len(eval_df):,}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("\nConfusion Matrix:")
    print("                Predicted Negative  Predicted Positive")
    print(f"True Negative   {tn:<20} {fp}")
    print(f"True Positive   {fn:<20} {tp}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Negative", "Positive"], zero_division=0))
    print("\nDetailed Metrics (Positive Class):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}" if auc is not None else "  ROC AUC:   N/A")

    return {
        "split": split_name,
        "threshold": threshold,
        "n_windows": int(len(eval_df)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": None if auc is None else float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    args = parse_args()
    print("=" * 70)
    print("SLIDING WINDOW EVALUATION - PIPELINE ALIGNED")
    print("=" * 70)

    model_path, metadata_path = resolve_paths(args)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_names = metadata.get("feature_names", [])
    threshold = args.threshold if args.threshold is not None else float(metadata.get("selected_threshold", 0.5))

    val_df = pd.read_csv(args.val_features)
    test_df = pd.read_csv(args.test_features)

    # Fallback for older metadata that may not include feature names.
    if not feature_names:
        feature_names = infer_feature_names(val_df)

    model = joblib.load(model_path)

    print(f"Model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Threshold source: {'CLI override' if args.threshold is not None else 'model metadata'}")
    print(f"Threshold value: {threshold:.3f}")
    print(f"Feature count: {len(feature_names)}")

    val_metrics = evaluate_split(model, val_df, feature_names, threshold, "validation")
    test_metrics = evaluate_split(model, test_df, feature_names, threshold, "test")

    os.makedirs(args.results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.results_dir, f"sliding_eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "metadata_path": metadata_path,
                "threshold": threshold,
                "validation": val_metrics,
                "test": test_metrics,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("SLIDING WINDOW EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
