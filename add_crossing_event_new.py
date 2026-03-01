#!/usr/bin/env python3
"""
Add 'crossing_event' variable to all .nc files in processed/.
Outputs modified files to new_processed/ with the same directory structure.

Label values written per timestep:
  0  — no crossing (default for all timesteps)
  1  — crossing event (BS or MP) (change_point from labels/bs/all/ or labels/mp/all/)

Exclusion:
  If a file's stem appears in labels/dg/ OR labels/sc/ it is skipped
  entirely — no output file is written.
  "None" class examples (label 0) come naturally from the timesteps in
  BS and MP files that fall outside the ±WINDOW_SECONDS crossing windows.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Paths — edit these to match your system
# ---------------------------------------------------------------------------

PROCESSED_DIR     = Path("./data/processed/2004")  # delete this later "2004"
NEW_PROCESSED_DIR = Path("./data/new_processed")

LABELS_ROOT = Path("./data/zenodo-3946033/crossings/labels")

BS_DIR = LABELS_ROOT / "bs" / "all"   # bow shock   → label 1
MP_DIR = LABELS_ROOT / "mp" / "all"   # magnetopause → label 2
DG_DIR = LABELS_ROOT / "dg"           # data gap    → exclude file
SC_DIR = LABELS_ROOT / "sc"           # s/c manoeuvre → exclude file

# ±window around each change_point (in seconds).
# At ~8 s cadence, 240 s ≈ 30 timesteps = ±4 minutes (matches roadmap Step 3).
WINDOW_SECONDS = 240


# ---------------------------------------------------------------------------
# YAML parsing  (no third-party yaml library needed)
# ---------------------------------------------------------------------------

def parse_change_points(yaml_path: Path) -> list[str]:
    """
    Read a YAML file and return the list of strings under change_points.
    Returns an empty list if the file does not exist or the field is absent.
    """
    if not yaml_path.exists():
        return []

    change_points = []
    inside_cp_block = False

    with open(yaml_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if stripped == "change_points:":
                inside_cp_block = True
                continue

            # Any top-level key (no leading spaces, ends with ':') closes the block
            if inside_cp_block:
                if stripped.startswith("- "):
                    change_points.append(stripped[2:].strip())
                elif not line.startswith(" ") and stripped.endswith(":"):
                    inside_cp_block = False

    return change_points


def yaml_exists(directory: Path, stem: str) -> bool:
    """Return True if <directory>/<stem>.yaml exists."""
    return (directory / f"{stem}.yaml").exists()


# ---------------------------------------------------------------------------
# Per-timestep label array construction
# ---------------------------------------------------------------------------

def build_label_array(
    time_array: np.ndarray,
    bs_yaml: Path,
    mp_yaml: Path,
) -> np.ndarray:
    """
    Build a per-timestep binary label array.

    Steps:
      1. Start with all zeros  (no crossing)
      2. Apply BS change_points → set window around each point to 1
      3. Apply MP change_points → set window around each point to 1

    Args:
        time_array : numpy array of datetime64 timestamps from the .nc file
        bs_yaml    : Path to the bow shock YAML (may not exist)
        mp_yaml    : Path to the magnetopause YAML (may not exist)

    Returns:
        numpy int8 array of shape (n_timesteps,) with values in {0, 1}
    """
    labels = np.zeros(len(time_array), dtype=np.int8)

    # Both BS and MP mark as 1 — binary: crossing vs no crossing
    for yaml_path, label_name in [(bs_yaml, "BS"), (mp_yaml, "MP")]:
        change_points = parse_change_points(yaml_path)
        for cp_str in change_points:
            if not cp_str:
                continue
            try:
                cp_dt = datetime.strptime(cp_str, "%d-%m-%Y/%H:%M:%S")
                cp_np = np.datetime64(cp_dt)

                diff_seconds = np.abs(
                    (time_array - cp_np) / np.timedelta64(1, "s")
                )
                window_mask = diff_seconds <= WINDOW_SECONDS
                labels[window_mask] = 1

                print(f"    {label_name} change_point {cp_str} → {window_mask.sum()} timesteps marked as 1")

            except Exception as e:
                print(f"  Warning: could not parse change_point '{cp_str}': {e}")

    return labels


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_nc_file(input_path: Path, output_path: Path, stem: str):
    """
    Load one .nc file, attach a crossing_event variable, and save.

    Args:
        input_path  : source .nc file
        output_path : destination .nc file (parent dirs created if needed)
        stem        : base filename without extension (used to look up YAMLs)
    """
    ds = xr.open_dataset(input_path)

    # --- Fix sentinel missing-data values in count_rate -------------------
    if "count_rate" in ds:
        cr = ds["count_rate"].values
        n_missing = np.sum(cr == 65535.0)
        if n_missing > 0:
            print(f"  Fixing {n_missing} sentinel values (65535 → NaN) in count_rate")
            cr[cr == 65535.0] = np.nan
            ds["count_rate"].values = cr

    # --- Guard: skip if crossing_event already present --------------------
    if "crossing_event" in ds:
        print(f"  Skipping — crossing_event already exists in {input_path.name}")
        ds.close()
        return

    # --- Build label array ------------------------------------------------
    time_array = ds["time"].values
    bs_yaml    = BS_DIR / f"{stem}.yaml"
    mp_yaml    = MP_DIR / f"{stem}.yaml"
    labels     = build_label_array(time_array, bs_yaml, mp_yaml)

    n_crossing = int((labels == 1).sum())
    n_none     = int((labels == 0).sum())
    print(f"  Labels — none: {n_none}, crossing (1): {n_crossing}")

    # --- Attach DataArray to dataset --------------------------------------
    ds["crossing_event"] = xr.DataArray(
        data=labels,
        dims=("time",),
        coords={"time": ds["time"]},
        attrs={
            "long_name": "Crossing Event Label",
            "description": (
                "Per-timestep binary crossing label. "
                "0 = no crossing, 1 = crossing (bow shock or magnetopause)."
            ),
            "units": "1",
            "flag_values": "0, 1",
            "flag_meanings": "no_crossing crossing",
            "window_seconds": WINDOW_SECONDS,
            "comment": "Derived from Zenodo YAML change_points (bs/all and mp/all)",
        },
    )

    # --- Save -------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(
        output_path,
        encoding={"crossing_event": {"zlib": True, "complevel": 4, "dtype": "int8"}},
    )
    print(f"  Saved → {output_path}")
    ds.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    nc_files = sorted(PROCESSED_DIR.rglob("*.nc"))

    print("=" * 80)
    print("ADDING crossing_event VARIABLE TO .NC FILES")
    print("=" * 80)
    print(f"Input  : {PROCESSED_DIR}")
    print(f"Output : {NEW_PROCESSED_DIR}")
    print(f"BS labels  : {BS_DIR}  → label 1 (crossing)")
    print(f"MP labels  : {MP_DIR}  → label 1 (crossing)")
    print(f"DG files   : {DG_DIR}  → excluded (bad data)")
    print(f"SC files   : {SC_DIR}  → excluded (bad data)")
    print(f"Window     : ±{WINDOW_SECONDS}s around each change_point")
    print(f"Note       : label 0 (none) comes from timesteps outside crossing windows")
    print(f"\nFound {len(nc_files)} .nc files")
    print("=" * 80)
    print()

    counts = {"processed": 0,"crossing": 0, "no-crossing": 0, "excluded": 0, "errors": 0}

    for i, nc_path in enumerate(nc_files, 1):
        stem        = nc_path.stem
        rel_path    = nc_path.relative_to(PROCESSED_DIR)
        output_path = NEW_PROCESSED_DIR / rel_path

        print(f"[{i}/{len(nc_files)}] {nc_path.name}")

        # dg/ or sc/ → skip entirely, data is unreliable
        if yaml_exists(DG_DIR, stem):
            print(f"  EXCLUDED — found in dg/ (data gap)")
            counts["excluded"] += 1
            continue

        if yaml_exists(SC_DIR, stem):
            print(f"  EXCLUDED — found in sc/ (spacecraft manoeuvre)")
            counts["excluded"] += 1
            continue

        if yaml_exists(BS_DIR, stem):
            counts["crossing"] += 1
        else:
            counts["no-crossing"] += 1

        if yaml_exists(MP_DIR, stem):
            counts["crossing"] += 1
        else:
            counts["no-crossing"] += 1

        try:
            process_nc_file(nc_path, output_path, stem)
            counts["processed"] += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            counts["errors"] += 1

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files found  : {len(nc_files)}")
    print(f"Processed          : {counts['processed']}")
    print(f"Crossing           : {counts['crossing']}")
    print(f"No-crossing        : {counts['no-crossing']}")
    print(f"Excluded (dg/sc)   : {counts['excluded']}")
    print(f"Errors             : {counts['errors']}")
    print(f"Output directory   : {NEW_PROCESSED_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
