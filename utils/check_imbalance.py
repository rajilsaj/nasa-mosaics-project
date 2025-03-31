import pandas as pd
import numpy as np
from tqdm import tqdm

def check_class_distribution(file_path, window_size=50):
    """Check class distribution in the data."""
    print("Loading data...", flush=True)
    df = pd.read_csv(file_path)
    
    # Check raw class distribution
    print("\nRaw class distribution:", flush=True)
    unique, counts = np.unique(df['gt_detection_win'], return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count:,} samples ({count/len(df)*100:.2f}%)", flush=True)
    
    # Check distribution in windows
    print(f"\nChecking distribution in {window_size}-sample windows...", flush=True)
    total_windows = len(df) - window_size
    window_labels = []
    
    for i in tqdm(range(total_windows), desc="Processing windows"):
        window_labels.append(df['gt_detection_win'][i])
    
    unique, counts = np.unique(window_labels, return_counts=True)
    print("\nWindow class distribution:", flush=True)
    for cls, count in zip(unique, counts):
        print(f"Class {cls}: {count:,} windows ({count/len(window_labels)*100:.2f}%)", flush=True)

if __name__ == "__main__":
    check_class_distribution("ml_ready_vortex_data.csv") 