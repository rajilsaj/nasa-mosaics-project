import pandas as pd
import numpy as np
from pathlib import Path

# === Configuration ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "ml_ready_vortex_data.csv"

# === Load Data ===
if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

print(f"📂 Reading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# === Identify Vortex Events ===
vortex_indices = df[df['gt_fwhm'] > 0].index
vortex_groups = []

if vortex_indices.empty:
    print("⚠️ No vortex events found in the dataset.")
else:
    current_group = [vortex_indices[0]]
    for i in range(1, len(vortex_indices)):
        if vortex_indices[i] - vortex_indices[i - 1] == 1:
            current_group.append(vortex_indices[i])
        else:
            vortex_groups.append(current_group)
            current_group = [vortex_indices[i]]
    vortex_groups.append(current_group)

    # === Overview ===
    print(f"\n🌪️ Total vortex samples: {len(vortex_indices)}")
    print(f"🧩 Number of unique vortex groups: {len(vortex_groups)}")

    # === Print First 5 Vortex Groups ===
    print("\n🔍 First 5 vortex groups:")
    for i, group in enumerate(vortex_groups[:5]):
        print(f"\nVortex {i + 1}:")
        print(f"   🟢 Start index: {group[0]}")
        print(f"   🔴 End index: {group[-1]}")
        print(f"   ⏱ Duration: {len(group)} samples")
        print(f"   🧭 Detection window values: {df['gt_detection_win'].iloc[max(group[0]-10, 0):group[-1]+1].values}")

    # === Duration Stats ===
    durations = [len(group) for group in vortex_groups]
    print("\n📊 Vortex Duration Statistics:")
    print(f"   📈 Average: {np.mean(durations):.2f} samples")
    print(f"   🔼 Max: {np.max(durations)} samples")
    print(f"   🔽 Min: {np.min(durations)} samples")
