#!/usr/bin/env python3
"""
AE-gated Random Forest pipeline with validation-first selection.

Modes:
1. validate (default): tune AE gating percentile + RF threshold on validation.
2. test: run frozen best config on test fixed + sliding sets.
3. legacy: preserve prior train/test-only behavior.
"""

import argparse
import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv1D, Input
from tensorflow.keras.optimizers import Adam

import warnings

warnings.filterwarnings("ignore")


class Config:
    """Configuration for AE gating experiment."""

    # Fixed-window features
    TRAIN_FEATURES_FILE = "train_features.csv"
    VAL_FEATURES_FILE = "val_features.csv"
    TEST_FEATURES_FILE = "test_features.csv"

    # Sliding-window features
    VAL_SLIDING_FEATURES_FILE = "val_sliding_features.csv"
    TEST_SLIDING_FEATURES_FILE = "test_sliding_features.csv"

    # Raw windows used for AE scoring
    TRAIN_WINDOWS_FILE = "train_windows.csv"
    VAL_WINDOWS_FILE = "val_windows.csv"

    # Output directories
    MODELS_DIR = "models"
    RESULTS_DIR = "results"

    # Autoencoder configuration (TCN-style Conv1D autoencoder)
    AE_MODEL_TYPE = "tcn"
    AE_WINDOW_SIZE = 60
    AE_EPOCHS = 40
    AE_BATCH_SIZE = 32
    AE_LEARNING_RATE = 1e-3
    AE_RANDOM_STATE = 42
    AE_TCN_FILTERS = 32
    AE_DILATIONS = [1, 2, 4, 8]
    AE_DROPOUT = 0.0

    # Gating search space
    GATING_PERCENTILES = [30, 40, 50, 60, 70]
    # Finer default grid for deployment-like threshold tuning.
    THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.10, 0.96, 0.01).tolist()]

    # Selection policy
    PRIMARY_METRIC = "f1_sliding_val"
    MIN_PRECISION = 0.0
    MIN_RECALL = 0.0

    # Random Forest parameters
    RF_PARAMS = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    # Feature columns to exclude
    EXCLUDE_COLUMNS = ["window_id", "event_sclk", "label"]
    EXCLUDE_COLUMNS_SLIDING = [
        "window_id",
        "start_idx",
        "end_idx",
        "start_sclk",
        "end_sclk",
        "label",
        "sliding_window_id",
        "sliding_start_idx",
        "sliding_end_idx",
        "sliding_start_sclk",
        "sliding_end_sclk",
    ]

    BASELINE_METRICS = {
        "precision": 0.0378,
        "recall": 0.0658,
        "f1_score": 0.0480,
        "roc_auc": 0.7457,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="AE-gated RF pipeline with validation")
    parser.add_argument(
        "--mode",
        choices=["validate", "test", "legacy"],
        default="validate",
        help="Pipeline mode. validate selects best config; test evaluates frozen config.",
    )
    parser.add_argument(
        "--best-config",
        default="",
        help="Path to best config json (required for --mode test unless auto-discovered).",
    )
    parser.add_argument(
        "--percentiles",
        default="",
        help="Comma-separated gating percentiles override, e.g. 30,50,70",
    )
    parser.add_argument(
        "--thresholds",
        default="",
        help="Comma-separated threshold override, e.g. 0.4,0.6,0.9",
    )
    parser.add_argument(
        "--primary-metric",
        default="",
        help=(
            "Override selection metric, e.g. f1_sliding_val, precision_sliding_val, "
            "recall_sliding_val, f1_fixed_val"
        ),
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=None,
        help="Override minimum sliding validation precision constraint.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=None,
        help="Override minimum sliding validation recall constraint.",
    )
    return parser.parse_args()


def parse_num_list(raw, cast=float):
    if not raw:
        return None
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def ensure_dirs():
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)


def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_window_groups(path, split_label):
    print("\n" + "=" * 70)
    print(f"LOADING {split_label.upper()} WINDOWS")
    print("=" * 70)
    if not os.path.exists(path):
        print(f"[ERROR] Window file not found: {path}")
        return None
    windows_df = pd.read_csv(path)
    print(f"Loaded {len(windows_df):,} rows")

    if "PRESSURE" not in windows_df.columns:
        print("[ERROR] PRESSURE column not found in windows file.")
        return None

    if "window_id" in windows_df.columns:
        groups = windows_df.groupby("window_id")
        print(f"Found {len(groups):,} unique windows")
        return groups

    print("[WARNING] window_id missing. Falling back to sequential 60-length windows.")
    groups = []
    for i in range(0, len(windows_df), 60):
        chunk = windows_df.iloc[i : i + 60]
        if len(chunk) == 60:
            groups.append((i // 60, chunk))
    return groups


def train_autoencoder(window_groups):
    print("\n" + "=" * 70)
    print("TRAINING AUTOENCODER")
    print("=" * 70)

    pressure_windows = []
    for _, window_data in window_groups:
        pressure = window_data["PRESSURE"].values
        if len(pressure) == 60:
            pressure_windows.append(pressure)

    if not pressure_windows:
        print("[ERROR] No valid 60-sample pressure windows found.")
        return None, None

    X_ae = np.array(pressure_windows)
    print(f"Extracted {len(X_ae):,} windows of size {X_ae.shape[1]}")

    scaler = StandardScaler()
    X_ae_scaled = scaler.fit_transform(X_ae)

    X_ae_scaled = X_ae_scaled.reshape((-1, Config.AE_WINDOW_SIZE, 1))

    # Residual dilated Conv1D stack (TCN-style block).
    def tcn_block(x, filters, dilation):
        residual = x
        y = Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(x)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)
        y = Conv1D(filters, 3, padding="causal", dilation_rate=dilation)(y)
        y = BatchNormalization()(y)
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, 1, padding="same")(residual)
        y = Add()([residual, y])
        return Activation("relu")(y)

    inp = Input(shape=(Config.AE_WINDOW_SIZE, 1))
    x = inp
    for d in Config.AE_DILATIONS:
        x = tcn_block(x, Config.AE_TCN_FILTERS, d)
    bottleneck = Conv1D(16, 1, padding="same", activation="relu")(x)
    x = bottleneck
    for d in reversed(Config.AE_DILATIONS):
        x = tcn_block(x, Config.AE_TCN_FILTERS, d)
    out = Conv1D(1, 1, padding="same", activation="linear")(x)

    autoencoder = Model(inputs=inp, outputs=out, name="tcn_autoencoder")
    autoencoder.compile(optimizer=Adam(learning_rate=Config.AE_LEARNING_RATE), loss="mse")

    start = time.time()
    autoencoder.fit(
        X_ae_scaled,
        X_ae_scaled,
        epochs=Config.AE_EPOCHS,
        batch_size=Config.AE_BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )
    elapsed = time.time() - start

    recon = autoencoder.predict(X_ae_scaled, verbose=0)
    err = np.mean((X_ae_scaled - recon) ** 2)
    print(f"AE training completed in {elapsed:.2f}s")
    print(f"AE mean reconstruction error: {err:.6f}")
    return autoencoder, scaler


def score_windows_with_ae(autoencoder, scaler, window_groups, split_name):
    print("\n" + "=" * 70)
    print(f"SCORING {split_name.upper()} WINDOWS WITH AUTOENCODER")
    print("=" * 70)
    scores = {}
    for window_id, window_data in window_groups:
        pressure = window_data["PRESSURE"].values
        if len(pressure) != 60:
            continue
        pressure_scaled = scaler.transform(pressure.reshape(1, -1)).reshape(
            (1, Config.AE_WINDOW_SIZE, 1)
        )
        recon = autoencoder.predict(pressure_scaled, verbose=0)
        mse = np.mean((pressure_scaled - recon) ** 2)
        scores[window_id] = float(mse)

    if not scores:
        print("[ERROR] No windows could be scored.")
        return {}

    vals = list(scores.values())
    print(f"Scored windows: {len(scores):,}")
    print(f"Score range: {min(vals):.6f} - {max(vals):.6f}")
    print(f"Mean score: {np.mean(vals):.6f}  Std: {np.std(vals):.6f}")
    return scores


def load_features_df(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {len(df):,} rows")
    return df


def normalize_labels(series):
    if series.dtype == object:
        return series.map({"True": 1, "False": 0}).fillna(series).astype(int)
    return series.astype(int)


def filter_training_data(train_df, window_scores, filter_percentile=50):
    """Apply AE gating to fixed-window train features."""
    if "window_id" not in train_df.columns:
        raise ValueError("window_id column missing in training features.")

    df = train_df.copy()
    df["ae_score"] = df["window_id"].map(window_scores)
    total = len(df)
    scored = int(df["ae_score"].notna().sum())
    if scored == 0:
        raise ValueError("No training rows matched AE window scores.")

    threshold = np.percentile(df["ae_score"].dropna(), 100 - filter_percentile)
    filtered = df[df["ae_score"] >= threshold].copy()
    filtered = filtered.drop(columns=["ae_score"])

    stats = {
        "total_rows": total,
        "scored_rows": scored,
        "coverage_pct": (scored / total) * 100.0,
        "filter_percentile": filter_percentile,
        "score_threshold": float(threshold),
        "kept_rows": len(filtered),
        "kept_pct": (len(filtered) / total) * 100.0,
    }

    if "label" in filtered.columns and len(filtered) > 0:
        y = normalize_labels(filtered["label"])
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        stats["class_distribution"] = {"positive": pos, "negative": neg}
    else:
        stats["class_distribution"] = {"positive": 0, "negative": 0}

    return filtered, stats


def prepare_fixed_Xy(df, feature_cols=None):
    y = normalize_labels(df["label"]).values
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in Config.EXCLUDE_COLUMNS]
    X = df[feature_cols].values
    return X, y, feature_cols


def prepare_sliding_Xy(df, feature_cols=None):
    valid = df[df["label"] != "Omit"].copy()
    valid["label"] = valid["label"].map({"True": 1, "False": 0})
    if feature_cols is None:
        feature_cols = [c for c in valid.columns if c not in Config.EXCLUDE_COLUMNS_SLIDING]
    X = valid[feature_cols].values
    y = valid["label"].astype(int).values
    return X, y, feature_cols


def train_rf_model(X_train, y_train):
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 70)
    print(f"Training rows: {len(X_train):,}")
    print(f"Class distribution: {np.bincount(y_train)}")
    print(f"RF params: {Config.RF_PARAMS}")
    start = time.time()
    model = RandomForestClassifier(**Config.RF_PARAMS)
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"RF training completed in {elapsed:.2f}s")
    return model, elapsed


def evaluate_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_threshold_grid(model, X, y, thresholds, split_name):
    y_prob = model.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_prob)
    rows = []
    for thr in thresholds:
        m = evaluate_at_threshold(y, y_prob, thr)
        m["roc_auc"] = float(roc_auc)
        m["split"] = split_name
        rows.append(m)
    return rows


def select_best_candidate(candidates):
    metric = Config.PRIMARY_METRIC
    if metric not in candidates[0]:
        print(
            f"[WARNING] PRIMARY_METRIC '{metric}' not found in candidates. "
            "Falling back to 'f1_sliding_val'."
        )
        metric = "f1_sliding_val"

    eligible = [
        c
        for c in candidates
        if c["precision_sliding_val"] >= Config.MIN_PRECISION
        and c["recall_sliding_val"] >= Config.MIN_RECALL
    ]
    pool = eligible if eligible else candidates
    pool = sorted(
        pool,
        key=lambda c: (
            c[metric],
            c["f1_sliding_val"],
            c["precision_sliding_val"],
            -c["threshold"],  # Prefer lower threshold on ties (recall-friendly)
            c["kept_rows"],
        ),
        reverse=True,
    )
    return pool[0]


def compare_to_baseline(new_metrics, baseline_metrics):
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'Baseline':<12} {'New Model':<12} {'Change':<18}")
    print("-" * 70)
    for metric in ["precision", "recall", "f1_score", "roc_auc"]:
        base = baseline_metrics.get(metric, 0.0)
        new = new_metrics.get(metric, 0.0)
        delta = new - base
        pct = (delta / base * 100.0) if base > 0 else 0.0
        print(f"{metric:<15} {base:<12.4f} {new:<12.4f} {delta:+.4f} ({pct:+.1f}%)")


def save_validation_outputs(
    timestamp,
    autoencoder,
    scaler,
    best_model,
    best_candidate,
    all_candidates,
    train_scores,
    val_scores,
):
    ensure_dirs()
    ae_model_path = os.path.join(Config.MODELS_DIR, f"ae_tcn_model_{timestamp}.keras")
    ae_scaler_path = os.path.join(Config.MODELS_DIR, f"ae_tcn_scaler_{timestamp}.pkl")
    rf_path = os.path.join(Config.MODELS_DIR, f"rf_ae_gated_ml_{timestamp}.pkl")
    autoencoder.save(ae_model_path)
    joblib.dump(scaler, ae_scaler_path)
    joblib.dump(best_model, rf_path)

    score_path = os.path.join(Config.RESULTS_DIR, f"ae_window_scores_{timestamp}.json")
    with open(score_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_scores, "val": val_scores}, f, indent=2)

    sweep_path = os.path.join(Config.RESULTS_DIR, f"ae_validation_sweep_{timestamp}.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "candidates": all_candidates}, f, indent=2)

    best_path = os.path.join(Config.RESULTS_DIR, f"ae_validation_best_config_{timestamp}.json")
    best_payload = {
        "timestamp": timestamp,
        "mode": "validate",
        "selected_config": best_candidate,
        "ae_model_path": ae_model_path,
        "ae_scaler_path": ae_scaler_path,
        "ae_model_type": Config.AE_MODEL_TYPE,
        "rf_model_path": rf_path,
        "threshold": best_candidate["threshold"],
        "filter_percentile": best_candidate["filter_percentile"],
        "feature_cols": best_candidate["feature_cols"],
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    return ae_model_path, rf_path, score_path, sweep_path, best_path


def run_validate_mode(percentiles, thresholds):
    print("=" * 70)
    print("AE GATING VALIDATION MODE")
    print("=" * 70)

    train_groups = load_window_groups(Config.TRAIN_WINDOWS_FILE, "train")
    val_groups = load_window_groups(Config.VAL_WINDOWS_FILE, "val")
    if train_groups is None or val_groups is None:
        return 1

    autoencoder, scaler = train_autoencoder(train_groups)
    if autoencoder is None:
        return 1

    train_scores = score_windows_with_ae(autoencoder, scaler, train_groups, "train")
    val_scores = score_windows_with_ae(autoencoder, scaler, val_groups, "val")
    if not train_scores:
        return 1

    train_df = load_features_df(Config.TRAIN_FEATURES_FILE, "train_features")
    val_fixed_df = load_features_df(Config.VAL_FEATURES_FILE, "val_features")
    val_sliding_df = load_features_df(Config.VAL_SLIDING_FEATURES_FILE, "val_sliding_features")

    all_candidates = []
    models_by_percentile = {}
    feature_cols_by_percentile = {}

    for pct in percentiles:
        print("\n" + "=" * 70)
        print(f"GATING PERCENTILE SWEEP: {pct}%")
        print("=" * 70)
        gated_train_df, gate_stats = filter_training_data(train_df, train_scores, filter_percentile=pct)
        if len(gated_train_df) < 20:
            print("[WARNING] Too few training rows after gating. Skipping.")
            continue

        X_train, y_train, feature_cols = prepare_fixed_Xy(gated_train_df)
        X_val_fixed, y_val_fixed, _ = prepare_fixed_Xy(val_fixed_df, feature_cols=feature_cols)
        X_val_sliding, y_val_sliding, _ = prepare_sliding_Xy(val_sliding_df, feature_cols=feature_cols)

        model, train_time = train_rf_model(X_train, y_train)
        models_by_percentile[pct] = model
        feature_cols_by_percentile[pct] = feature_cols

        fixed_rows = evaluate_threshold_grid(model, X_val_fixed, y_val_fixed, thresholds, "val_fixed")
        sliding_rows = evaluate_threshold_grid(model, X_val_sliding, y_val_sliding, thresholds, "val_sliding")
        fixed_by_thr = {r["threshold"]: r for r in fixed_rows}

        for row in sliding_rows:
            thr = row["threshold"]
            f = fixed_by_thr[thr]
            all_candidates.append(
                {
                    "filter_percentile": int(pct),
                    "threshold": float(thr),
                    "training_time_seconds": float(train_time),
                    "kept_rows": int(gate_stats["kept_rows"]),
                    "coverage_pct": float(gate_stats["coverage_pct"]),
                    "score_threshold": float(gate_stats["score_threshold"]),
                    "precision_sliding_val": row["precision"],
                    "recall_sliding_val": row["recall"],
                    "f1_sliding_val": row["f1_score"],
                    "roc_auc_sliding_val": row["roc_auc"],
                    "precision_fixed_val": f["precision"],
                    "recall_fixed_val": f["recall"],
                    "f1_fixed_val": f["f1_score"],
                    "roc_auc_fixed_val": f["roc_auc"],
                    "feature_cols": feature_cols,
                }
            )

    if not all_candidates:
        print("[ERROR] No valid candidates were produced during validation sweep.")
        return 1

    best = select_best_candidate(all_candidates)
    best_model = models_by_percentile[best["filter_percentile"]]
    ts = now_ts()
    ae_path, rf_path, score_path, sweep_path, best_path = save_validation_outputs(
        ts, autoencoder, scaler, best_model, best, all_candidates, train_scores, val_scores
    )

    print("\n" + "=" * 70)
    print("VALIDATION SELECTION COMPLETED")
    print("=" * 70)
    print(f"Selected percentile: {best['filter_percentile']}")
    print(f"Selected threshold:  {best['threshold']:.2f}")
    print(
        f"Sliding val metrics: P={best['precision_sliding_val']:.4f} "
        f"R={best['recall_sliding_val']:.4f} F1={best['f1_sliding_val']:.4f}"
    )
    print(f"Saved AE model: {ae_path}")
    print(f"Saved RF model: {rf_path}")
    print(f"Saved scores:   {score_path}")
    print(f"Saved sweep:    {sweep_path}")
    print(f"Saved best cfg: {best_path}")
    return 0


def discover_latest_best_config():
    if not os.path.exists(Config.RESULTS_DIR):
        return ""
    candidates = [
        os.path.join(Config.RESULTS_DIR, x)
        for x in os.listdir(Config.RESULTS_DIR)
        if x.startswith("ae_validation_best_config_") and x.endswith(".json")
    ]
    if not candidates:
        return ""
    return max(candidates, key=os.path.getctime)


def run_test_mode(best_config_path):
    print("=" * 70)
    print("AE GATING TEST MODE")
    print("=" * 70)
    if not best_config_path:
        best_config_path = discover_latest_best_config()
    if not best_config_path or not os.path.exists(best_config_path):
        print("[ERROR] best config json not found. Run --mode validate first.")
        return 1

    with open(best_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    rf_model_path = cfg["rf_model_path"]
    threshold = float(cfg["threshold"])
    feature_cols = cfg["feature_cols"]
    if not os.path.exists(rf_model_path):
        print(f"[ERROR] RF model not found: {rf_model_path}")
        return 1

    model = joblib.load(rf_model_path)
    test_fixed_df = load_features_df(Config.TEST_FEATURES_FILE, "test_features")
    test_sliding_df = load_features_df(Config.TEST_SLIDING_FEATURES_FILE, "test_sliding_features")

    X_fixed, y_fixed, _ = prepare_fixed_Xy(test_fixed_df, feature_cols=feature_cols)
    X_sliding, y_sliding, _ = prepare_sliding_Xy(test_sliding_df, feature_cols=feature_cols)

    fixed_prob = model.predict_proba(X_fixed)[:, 1]
    sliding_prob = model.predict_proba(X_sliding)[:, 1]
    fixed_metrics = evaluate_at_threshold(y_fixed, fixed_prob, threshold)
    sliding_metrics = evaluate_at_threshold(y_sliding, sliding_prob, threshold)
    fixed_metrics["roc_auc"] = float(roc_auc_score(y_fixed, fixed_prob))
    sliding_metrics["roc_auc"] = float(roc_auc_score(y_sliding, sliding_prob))

    print("\nFixed test metrics:")
    print(
        f"  P={fixed_metrics['precision']:.4f} R={fixed_metrics['recall']:.4f} "
        f"F1={fixed_metrics['f1_score']:.4f} AUC={fixed_metrics['roc_auc']:.4f}"
    )
    print("\nSliding test metrics:")
    print(
        f"  P={sliding_metrics['precision']:.4f} R={sliding_metrics['recall']:.4f} "
        f"F1={sliding_metrics['f1_score']:.4f} AUC={sliding_metrics['roc_auc']:.4f}"
    )

    compare_to_baseline(sliding_metrics, Config.BASELINE_METRICS)

    ts = now_ts()
    report = {
        "timestamp": ts,
        "mode": "test",
        "best_config_path": best_config_path,
        "rf_model_path": rf_model_path,
        "threshold": threshold,
        "fixed_test_metrics": fixed_metrics,
        "sliding_test_metrics": sliding_metrics,
        "baseline_metrics": Config.BASELINE_METRICS,
    }
    out_path = os.path.join(Config.RESULTS_DIR, f"ae_final_test_report_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved final test report: {out_path}")
    return 0


def run_legacy_mode():
    """Preserve old train+test flow for backwards compatibility."""
    print("=" * 70)
    print("LEGACY MODE (TRAIN+TEST)")
    print("=" * 70)
    train_groups = load_window_groups(Config.TRAIN_WINDOWS_FILE, "train")
    if train_groups is None:
        return 1
    autoencoder, scaler = train_autoencoder(train_groups)
    if autoencoder is None:
        return 1
    train_scores = score_windows_with_ae(autoencoder, scaler, train_groups, "train")
    train_df = load_features_df(Config.TRAIN_FEATURES_FILE, "train_features")
    test_df = load_features_df(Config.TEST_FEATURES_FILE, "test_features")
    gated_train_df, gate_stats = filter_training_data(
        train_df, train_scores, filter_percentile=Config.GATING_PERCENTILES[2]
    )
    print(f"Gating kept rows: {gate_stats['kept_rows']} ({gate_stats['kept_pct']:.1f}%)")
    X_train, y_train, feature_cols = prepare_fixed_Xy(gated_train_df)
    X_test, y_test, _ = prepare_fixed_Xy(test_df, feature_cols=feature_cols)
    model, _ = train_rf_model(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_at_threshold(y_test, y_prob, 0.5)
    metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    compare_to_baseline(metrics, Config.BASELINE_METRICS)
    return 0


def main():
    args = parse_args()
    ensure_dirs()
    if args.primary_metric:
        Config.PRIMARY_METRIC = args.primary_metric
    if args.min_precision is not None:
        Config.MIN_PRECISION = float(args.min_precision)
    if args.min_recall is not None:
        Config.MIN_RECALL = float(args.min_recall)

    percentiles = parse_num_list(args.percentiles, cast=int) or Config.GATING_PERCENTILES
    thresholds = parse_num_list(args.thresholds, cast=float) or Config.THRESHOLD_GRID
    print(f"Run mode: {args.mode}")
    print(f"Gating percentiles: {percentiles}")
    print(f"Threshold grid: {thresholds}")
    print(f"Primary metric: {Config.PRIMARY_METRIC}")
    print(f"Min precision constraint: {Config.MIN_PRECISION}")
    print(f"Min recall constraint: {Config.MIN_RECALL}")

    if args.mode == "validate":
        return run_validate_mode(percentiles, thresholds)
    if args.mode == "test":
        return run_test_mode(args.best_config)
    return run_legacy_mode()


if __name__ == "__main__":
    raise SystemExit(main())
