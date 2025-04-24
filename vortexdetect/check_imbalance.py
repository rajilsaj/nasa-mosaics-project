import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def check_class_distribution(file_path: Path, window_size: int = 50):
    """Check class distribution in raw labels and sliding windows."""
    if not file_path.exists():
        raise FileNotFoundError(f" File not found: {file_path}")

    print(f" Loading data from {file_path.name}...", flush=True)
    df = pd.read_csv(file_path)

    if "gt_detection_win" not in df.columns:
        raise ValueError(" Column 'gt_detection_win' not found in the dataset.")

    # Raw class distribution
    print("\n Raw Label Distribution:", flush=True)
    unique, counts = np.unique(df['gt_detection_win'], return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count:,} samples ({count/len(df)*100:.2f}%)", flush=True)

    # Sliding window label distribution
    print(f"\n Checking distribution in {window_size}-sample windows...", flush=True)
    total_windows = len(df) - window_size
    window_labels = []

    for i in tqdm(range(total_windows), desc="Processing windows"):
        label = df['gt_detection_win'].iloc[i + window_size - 1]  # label at end of window
        window_labels.append(label)

    unique, counts = np.unique(window_labels, return_counts=True)
    print("\n Window Label Distribution:", flush=True)
    for cls, count in zip(unique, counts):
        print(f"   Class {cls}: {count:,} windows ({count/len(window_labels)*100:.2f}%)", flush=True)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_file = BASE_DIR / "data" / "ml_ready_vortex_data.csv"
    check_class_distribution(data_file)
