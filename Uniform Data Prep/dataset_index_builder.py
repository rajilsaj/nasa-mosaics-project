import os
import pandas as pd
from glob import glob

# -------- UPDATE THESE 3 PATHS --------
ELS_DIR = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\2004"
BASE_LABEL_DIR = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\zendo-3946033\crossings\labels"
OUTPUT_CSV = r"C:\Users\PC\Documents\Github\nasa-mosaics-project\data\dataset_index\dataset_index_2004.csv"
# -------------------------------------

BS_DIR = os.path.join(BASE_LABEL_DIR, "bs", "all")
MP_DIR = os.path.join(BASE_LABEL_DIR, "mp", "all")
SC_DIR = os.path.join(BASE_LABEL_DIR, "sc")

def yaml_keys(folder):
    return set(
        os.path.basename(f).replace(".yaml", "")
        for f in glob(os.path.join(folder, "*.yaml"))
    )

# Load label keys (Zenodo contains 2004–2012; intersection handles 2004 only)
bs_keys = yaml_keys(BS_DIR)
mp_keys = yaml_keys(MP_DIR)
sc_keys = yaml_keys(SC_DIR)

# Recursively find all 2004 raw CSVs
els_files = glob(os.path.join(ELS_DIR, "**", "*_raw.csv"), recursive=True)

rows = []
for f in els_files:
    filename = os.path.basename(f)
    file_key = filename.replace("_raw.csv", "")  # e.g., ELS_200417906_V01

    # Extract DOY from key: ELS_2004 + DOY(3) + ...  -> positions 8:11
    # Example: ELS_200417906_V01 -> doy = 179
    doy = int(file_key[8:11])

    has_bs = file_key in bs_keys
    has_mp = file_key in mp_keys
    has_sc = file_key in sc_keys

    # Exclusion rule: drop SC entirely
    exclude = has_sc

    # Split rule (roadmap)
    if doy <= 270:
        split = "train"
    elif doy <= 335:
        split = "val"
    else:
        split = "test"

    # Label type (for reporting)
    if has_bs and has_mp:
        label_type = "BS+MP"
    elif has_bs:
        label_type = "BS"
    elif has_mp:
        label_type = "MP"
    else:
        label_type = "none"

    rows.append({
        "file_key": file_key,
        "file_path": f,
        "doy": doy,
        "has_bs": has_bs,
        "has_mp": has_mp,
        "has_sc": has_sc,
        "exclude": exclude,
        "label_type": label_type,
        "split": split,
    })

df = pd.DataFrame(rows).sort_values(["doy", "file_key"]).reset_index(drop=True)

# Save
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

# Print summary
total = len(df)
excluded = int(df["exclude"].sum())
clean = total - excluded

print("Saved:", OUTPUT_CSV)
print("Total 2004 files:", total)
print("Excluded (SC):", excluded)
print("Clean files:", clean)

print("\nClean label breakdown:")
clean_df = df[df["exclude"] == False]
print(clean_df["label_type"].value_counts())

print("\nClean split breakdown:")
print(clean_df["split"].value_counts())