import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('ml_ready_vortex_data.csv')

# Find vortex events (consecutive TRUE values in gt_fwhm)
vortex_events = df[df['gt_fwhm'] > 0].index
vortex_groups = []
current_group = [vortex_events[0]]

for i in range(1, len(vortex_events)):
    if vortex_events[i] - vortex_events[i-1] == 1:
        current_group.append(vortex_events[i])
    else:
        vortex_groups.append(current_group)
        current_group = [vortex_events[i]]
vortex_groups.append(current_group)

print(f"Total number of vortex events: {len(vortex_events)}")
print(f"Number of unique vortices: {len(vortex_groups)}")
print("\nFirst 5 vortex groups:")
for i, group in enumerate(vortex_groups[:5]):
    print(f"\nVortex {i+1}:")
    print(f"Start index: {group[0]}")
    print(f"End index: {group[-1]}")
    print(f"Duration: {len(group)} samples")
    print(f"Associated detection windows: {df['gt_detection_win'].iloc[group[0]-10:group[-1]+1].values}")

# Calculate statistics
durations = [len(group) for group in vortex_groups]
print(f"\nVortex Statistics:")
print(f"Average duration: {np.mean(durations):.2f} samples")
print(f"Max duration: {np.max(durations)} samples")
print(f"Min duration: {np.min(durations)} samples") 