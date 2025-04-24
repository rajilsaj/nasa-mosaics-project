import pandas as pd
import numpy as np
from pathlib import Path

# === Configuration ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_vortex_data.csv"

# === Load Data ===
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# === Identify Indices ===
vortex_indices = df[df["gt_fwhm"] > 0].index
detection_indices = df[df["gt_detection_win"] > 0].index

print(f"\nTotal vortex events: {len(vortex_indices)}")
print(f"Total detection windows: {len(detection_indices)}")

# === Inspect First 10 Events ===
print("\nFirst 10 vortex events and their preceding detection windows:")
for i in range(min(10, len(vortex_indices))):
    vortex_idx = vortex_indices[i]
    preceding = detection_indices[detection_indices < vortex_idx]
    print(f"\nVortex at index: {vortex_idx}")
    print(f"   Preceding detection count: {len(preceding)}")
    if len(preceding) > 0:
        print(f"   Last detection before vortex: {preceding[-1]}")
        print(f"   Distance from vortex: {vortex_idx - preceding[-1]} samples")

# === Summary Stats ===
detection_counts = [len(detection_indices[detection_indices < idx]) for idx in vortex_indices]

print(f"\nDetection stats per vortex:")
print(f"   Average: {np.mean(detection_counts):.2f}")
print(f"   Max: {np.max(detection_counts)}")
print(f"   Min: {np.min(detection_counts)}")
