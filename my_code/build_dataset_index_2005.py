import os, glob
import pandas as pd

ELS_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/2005"
BASE_LABEL_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/zenodo5004160/zenodo-3946033/labels/labels"
OUTPUT_CSV = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/dataset_index_2005.csv"


rows = []
els_files = sorted(glob.glob(os.path.join(ELS_DIR, "**/*_raw.csv")))

for els_path in els_files:
    base = os.path.basename(els_path).replace("_raw.csv", "")

    mp_all = os.path.join(BASE_LABEL_DIR, "mp", "all", base + ".yaml")
    bs_all = os.path.join(BASE_LABEL_DIR, "bs", "all", base + ".yaml")
    dg = os.path.join(BASE_LABEL_DIR, "dg", base + ".yaml")
    sc = os.path.join(BASE_LABEL_DIR, "sc", base + ".yaml")

    has_mp = os.path.exists(mp_all)
    has_bs = os.path.exists(bs_all)
    has_dg = os.path.exists(dg)
    has_sc = os.path.exists(sc)

    rows.append({
        "els_file": els_path,
        "base": base,
        "has_mp": has_mp,
        "has_bs": has_bs,
        "has_dg": has_dg,
        "has_sc": has_sc,
        "exclude": has_dg or has_sc
    })

df = pd.DataFrame(rows)
df.to_csv("dataset_index_2005.csv", index=False)

print("ELS files:", len(df))
print("Counts:")
print(df[["has_mp","has_bs","has_dg","has_sc","exclude"]].sum())
print("\nNon-excluded with MP:", ((~df["exclude"]) & (df["has_mp"])).sum())
print("Non-excluded with BS:", ((~df["exclude"]) & (df["has_bs"])).sum())

