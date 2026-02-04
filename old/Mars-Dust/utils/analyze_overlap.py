import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/ml_ready_vortex_data.csv')

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

# --- NEW: Calculate average, max, and min length of contiguous vortex events ---
vortex_mask = (df['gt_detection_win'] == 1) | (df['gt_fwhm'] == 1)
event_lengths = []
current_length = 0
for is_vortex in vortex_mask:
    if is_vortex:
        current_length += 1
    elif current_length > 0:
        event_lengths.append(current_length)
        current_length = 0
if current_length > 0:
    event_lengths.append(current_length)

if event_lengths:
    print(f"\nAverage vortex event length: {np.mean(event_lengths):.2f} samples")
    print(f"Max vortex event length: {np.max(event_lengths)} samples")
    print(f"Min vortex event length: {np.min(event_lengths)} samples")
    print(f"Number of vortex events: {len(event_lengths)}")
else:
    print("\nNo vortex events found in the data.")
# --- END NEW --- 