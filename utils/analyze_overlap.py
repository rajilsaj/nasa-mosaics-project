import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('ml_ready_vortex_data.csv')

# Get indices of vortex events and detection windows
vortex_indices = df[df['gt_fwhm'] > 0].index
detection_indices = df[df['gt_detection_win'] > 0].index

print(f"Total vortex events: {len(vortex_indices)}")
print(f"Total detection windows: {len(detection_indices)}")

# Look at first 10 vortex events and their preceding detection windows
print("\nLooking at first 10 vortex events and their preceding detection windows:")
for i in range(min(10, len(vortex_indices))):
    vortex_idx = vortex_indices[i]
    preceding = detection_indices[detection_indices < vortex_idx]
    print(f"\nVortex at index {vortex_idx}")
    print(f"Number of preceding detection windows: {len(preceding)}")
    if len(preceding) > 0:
        print(f"Last detection window before vortex: {preceding[-1]}")
        print(f"Distance from vortex: {vortex_idx - preceding[-1]} samples")

# Calculate average number of detection windows per vortex
detection_counts = []
for vortex_idx in vortex_indices:
    preceding = detection_indices[detection_indices < vortex_idx]
    detection_counts.append(len(preceding))

print(f"\nAverage number of detection windows per vortex: {np.mean(detection_counts):.2f}")
print(f"Max number of detection windows per vortex: {np.max(detection_counts)}")
print(f"Min number of detection windows per vortex: {np.min(detection_counts)}") 