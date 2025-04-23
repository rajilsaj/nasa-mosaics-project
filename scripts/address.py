import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# ==== Config ====
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Parameters
DROP_THRESHOLD = 0.01
Z_SCORE_THRESHOLD = 1.0
SUB_WINDOW_SIZE = 10
STEP_SIZE = 1

def load_data(vortex_path: Path, data_path: Path):
    vortex_df = pd.read_csv(vortex_path)
    ml_df = pd.read_csv(data_path)

    for col in ["SCLK", "PRESSURE", "gt_detection_win", "gt_fwhm"]:
        if col not in ml_df.columns:
            raise ValueError(f"Missing column '{col}' in ml data.")

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
    ema = full_window["PRESSURE"].ewm(span=max(len(full_window) // 2, 1)).mean().iloc[-1]

    return {
        "mean_pressure": mean,
        "std_pressure": std,
        "pressure_change": change,
        "pressure_drop_ratio": drop_ratio,
        "z_score": z_score,
        "scheme1_detection": drop_ratio >= DROP_THRESHOLD,
        "scheme2_detection": z_score >= Z_SCORE_THRESHOLD,
        "long_term_mean": long_mean,
        "ema_pressure": ema,
        "trend": mean - long_mean,
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

def threshold_tuning(labeled_df, output_path: Path, model_name: str):
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

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1s, label="F1", marker="o")
    plt.plot(thresholds, precisions, label="Precision", marker="x")
    plt.plot(thresholds, recalls, label="Recall", marker="s")
    plt.axvline(best_thresh, linestyle="--", color="gray", label=f"Best: {best_thresh:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Tuning — {model_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = DATA_DIR / f"threshold_tuning_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()

    # Append threshold to history log
    with open(LOG_DIR / "threshold_history.csv", "a") as f:
        f.write(f"{model_name},{fixed_before}\n")

    print(f"[✓] Best Threshold: {best_thresh:.4f} for {model_name} saved to: {fig_path}")


def main(model_name="Random Forest", window_size=50):
    print("📥 Loading and preparing data...")
    vortex_file = DATA_DIR / "Jackson_vortex_detections_reformatted_augmented.csv"
    ml_data_file = DATA_DIR / "ml_ready_vortex_data.csv"
    output_path = DATA_DIR / "address.csv"

    sclk_list, ml_df = load_data(vortex_file, ml_data_file)

    print("🧠 Generating labeled windows...")
    labeled_df = generate_labeled_windows(sclk_list, ml_df, fixed_before=window_size)
    labeled_df.to_csv(output_path, index=False)
    print(f"[✓] Saved labeled windows to {output_path} — {len(labeled_df)} rows")

    print("📊 Performing threshold tuning...")
    threshold_tuning(labeled_df, output_path, model_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Random Forest", help="Model name (used for labeling plots)")
    parser.add_argument("--window", type=int, default=50, help="Window size used in sliding window")
    args = parser.parse_args()

    main(model_name=args.model, window_size=args.window)
