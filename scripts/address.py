import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score

# Parameters (defaults)
DROP_THRESHOLD = 0.01
Z_SCORE_THRESHOLD = 1.0
SUB_WINDOW_SIZE = 10
STEP_SIZE = 1

def load_data(vortex_path, data_path):
    vortex_df = pd.read_csv(vortex_path)
    ml_df = pd.read_csv(data_path)

    for col in ["SCLK", "PRESSURE", "gt_detection_win", "gt_fwhm"]:
        if col not in ml_df.columns:
            raise ValueError(f"Missing required column: {col}")

    ml_df["gt_detection_win"] = ml_df["gt_detection_win"].astype(bool)
    ml_df["gt_fwhm"] = ml_df["gt_fwhm"].astype(bool)

    return vortex_df["SCLK"].tolist(), ml_df


def extract_features(sub_window, full_window):
    initial = sub_window["PRESSURE"].iloc[0]
    final = sub_window["PRESSURE"].iloc[-1]
    mean = sub_window["PRESSURE"].mean()
    std = sub_window["PRESSURE"].std()
    change = final - initial
    drop_ratio = (initial - final) / initial if initial != 0 else 0
    z_score = (initial - final) / std if std != 0 else 0

    long_mean = full_window["PRESSURE"].mean()
    long_std = full_window["PRESSURE"].std()
    ema = full_window["PRESSURE"].ewm(span=max(len(full_window) // 2, 1)).mean().iloc[-1]
    trend = mean - long_mean

    return {
        "mean_pressure": mean,
        "std_pressure": std,
        "pressure_change": change,
        "pressure_drop_ratio": drop_ratio,
        "z_score": z_score,
        "scheme1_detection": drop_ratio >= DROP_THRESHOLD,
        "scheme2_detection": z_score >= Z_SCORE_THRESHOLD,
        "long_term_mean": long_mean,
        "long_term_std": long_std,
        "ema_pressure": ema,
        "trend": trend,
    }


def generate_labeled_windows(vortex_sclk_list, ml_df, fixed_before):
    labeled_rows = []

    for sclk in vortex_sclk_list:
        row_match = ml_df[ml_df["SCLK"] == sclk]
        if row_match.empty:
            continue

        idx = row_match.index[0]
        fixed_start = max(idx - fixed_before, 0)
        fixed_window = ml_df.iloc[fixed_start: idx]

        for i in range(0, len(fixed_window) - SUB_WINDOW_SIZE + 1, STEP_SIZE):
            sub = fixed_window.iloc[i:i + SUB_WINDOW_SIZE]
            r = sub.iloc[-1]

            if r["gt_detection_win"]:
                label = True
            elif not r["gt_detection_win"] and not r["gt_fwhm"]:
                label = False
            else:
                continue

            row_data = r.to_dict()
            row_data.update(extract_features(sub, fixed_window.iloc[:i + SUB_WINDOW_SIZE]))
            row_data.update({
                "sub_window_start_index": sub.index[0],
                "sub_window_end_index": sub.index[-1],
                "ml_label": label,
                "vortex_sclk": sclk
            })

            labeled_rows.append(row_data)

    return pd.DataFrame(labeled_rows)


def threshold_tuning(labeled_df, output_dir, model_name, window_size):
    y_true = labeled_df["ml_label"].astype(int)
    y_scores = labeled_df["pressure_drop_ratio"].values

    thresholds = np.linspace(y_scores.min(), y_scores.max(), 101)
    best_f1, best_thresh = 0, 0
    f1s, precisions, recalls = [], [], []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        f1s.append(f1)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"\n✅ Best threshold: {best_thresh:.4f}")
    print(f"F1 Score: {best_f1:.3f}")
    print(f"Precision: {precisions[np.argmax(f1s)]:.3f}")
    print(f"Recall: {recalls[np.argmax(f1s)]:.3f}")

    plt.figure()
    plt.plot(thresholds, f1s, label="F1", marker="o")
    plt.plot(thresholds, precisions, label="Precision", marker="x")
    plt.plot(thresholds, recalls, label="Recall", marker="s")
    plt.axvline(best_thresh, linestyle="--", color="gray")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.title(f"Threshold Tuning - {model_name}")
    plt.tight_layout()
    out_path = Path(output_dir) / f"threshold_tuning_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(out_path)
    plt.close()

    # Log the threshold setting
    log_path = Path("logs") / "threshold_history.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{model_name},{window_size}\n")


def main(model_name="Random Forest", window_size=100):
    vortex_file = Path("data/Jackson_vortex_detections_reformatted_augmented.csv")
    ml_data_file = Path("data/ml_ready_vortex_data.csv")
    output_path = Path("data/address.csv")

    print("📥 Loading and preparing data...")
    sclk_list, ml_df = load_data(vortex_file, ml_data_file)

    print("🧠 Generating labeled windows...")
    labeled_df = generate_labeled_windows(sclk_list, ml_df, fixed_before=window_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(output_path, index=False)
    print(f"[✓] Saved labeled windows to {output_path} — {len(labeled_df)} rows")

    print("📊 Performing threshold tuning...")
    threshold_tuning(labeled_df, "data", model_name, window_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Random Forest", help="Model name for labeling")
    parser.add_argument("--window", type=int, default=100, help="Sliding window size")
    args = parser.parse_args()

    main(model_name=args.model, window_size=args.window)
