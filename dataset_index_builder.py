import os
from glob import glob
from pathlib import Path

import pandas as pd

# ===== PATHS =====
ELS_DIR = r"YOUR PATH"
BASE_LABEL_DIR = r"YOUR PATH"
OUTPUT_CSV = r"YOUR PATH"
# =================

BS_DIR = os.path.join(BASE_LABEL_DIR, "bs", "all")
MP_DIR = os.path.join(BASE_LABEL_DIR, "mp", "all")
DG_DIR = os.path.join(BASE_LABEL_DIR, "dg")
SC_DIR = os.path.join(BASE_LABEL_DIR, "sc")


def yaml_keys(folder: str) -> set[str]:
    keys = set()
    for pattern in ("*.yaml", "*.yml"):
        for path in glob(os.path.join(folder, pattern)):
            keys.add(Path(path).stem)
    return keys


def get_doy(file_key: str) -> int:
    # ELS_200417906_V01 -> 179
    return int(file_key[8:11])


def get_split(doy: int) -> str:
    if doy <= 270:
        return "train"
    if doy <= 335:
        return "val"
    return "test"


def get_label_type(has_bs: bool, has_mp: bool) -> str:
    if has_bs and has_mp:
        return "BS+MP"
    if has_bs:
        return "BS"
    if has_mp:
        return "MP"
    return "none"


def main() -> None:
    bs_keys = yaml_keys(BS_DIR)
    mp_keys = yaml_keys(MP_DIR)
    dg_keys = yaml_keys(DG_DIR)
    sc_keys = yaml_keys(SC_DIR)

    els_files = sorted(glob(os.path.join(ELS_DIR, "**", "*_raw.csv"), recursive=True))

    rows = []
    for file_path in els_files:
        filename = os.path.basename(file_path)
        file_key = filename.replace("_raw.csv", "")
        doy = get_doy(file_key)

        has_bs = file_key in bs_keys
        has_mp = file_key in mp_keys
        has_dg = file_key in dg_keys
        has_sc = file_key in sc_keys
        exclude = has_dg or has_sc

        rows.append(
            {
                "file_key": file_key,
                "file_path": file_path,
                "doy": doy,
                "has_bs": has_bs,
                "has_mp": has_mp,
                "has_dg": has_dg,
                "has_sc": has_sc,
                "exclude": exclude,
                "label_type": get_label_type(has_bs, has_mp),
                "split": get_split(doy),
            }
        )

    df = pd.DataFrame(rows).sort_values(["doy", "file_key"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    clean = df.loc[~df["exclude"]].copy()
    print("Saved:", OUTPUT_CSV)
    print("Total files:", len(df))
    print(f"Excluded (DG or SC): {int(df['exclude'].sum())}")
    print("Clean files:", len(clean))
    print("\nLabel breakdown (clean only):")
    print(clean["label_type"].value_counts(dropna=False))
    print("\nSplit breakdown (clean only):")
    print(clean["split"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
