#!/usr/bin/env python3
from pathlib import Path
from pds4_tools import pds4_read
import xarray as xr
import numpy as np
import pandas as pd


def _to_np(field):
    """Ensure a plain NumPy array (avoid object arrays from pds4_tools)."""
    return np.array(field, dtype=float)


def _broadcast_if_1d(arr, target_len):
    """
    If arr is 1D (energy or anode definition is static),
    broadcast to (target_len, N). Otherwise return as-is.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return np.tile(arr, (target_len, 1))
    return arr


def process_file(xml_path):
    """
    Convert a PDS4 XML/DAT file pair into a rich NetCDF (xarray.Dataset)
    that preserves *all* metadata needed to reproduce NASA plotting.

    Returns
    -------
    xr.Dataset with:
      - dims: time, energy, anode
      - variables:
          count_rate(time, energy, anode)
          record_dur(time)                     # 'DT'
          energy_center(time, energy)          # GROUP_2, DIM1_E
          energy_lower(time, energy)           # GROUP_4, DIM1_E_LOWER
          energy_upper(time, energy)           # GROUP_3, DIM1_E_UPPER
          theta_center(time, anode)            # GROUP_5, DIM2_THETA
          theta_lower(time, anode)             # GROUP_7, DIM2_THETA_LOWER
          theta_upper(time, anode)             # GROUP_6, DIM2_THETA_UPPER
      - coords:
          time(datetime64[ns]), energy(index 0..62), anode(1..8)
    """
    xml_path = Path(xml_path)
    print(f"Processing {xml_path}")

    # 1) Read PDS4
    structure = pds4_read(str(xml_path), quiet=True)
    table = structure[0]
    data = table.data

    # 2) Time and duration
    time = pd.to_datetime(
        np.array([str(s) for s in data["UTC"]]),
        format="%Y-%jT%H:%M:%S.%f",
        errors="raise",
    ).to_numpy()
    nrec = time.shape[0]

    record_dur = _to_np(data["DT"])  # seconds; shape (nrec,)

    # 3) Counts → reshape to (time, energy=63, anode=8)
    raw_counts = _to_np(data["GROUP_1, DATA"]).reshape(nrec, 63, 8)

    # Replace sentinel overflow with NaN (counts and energies)
    raw_counts[raw_counts >= 65535.0] = np.nan

    # 4) Energy centers and edges (may be 1D or 2D in files)
    # NOTE: Your previous script did `[...] [0]` which dropped per-record variation.
    e_center = _to_np(data["GROUP_2, DIM1_E"])
    e_upper  = _to_np(data["GROUP_3, DIM1_E_UPPER"])
    e_lower  = _to_np(data["GROUP_4, DIM1_E_LOWER"])

    # Energy arrays can be (63,) or (nrec, 63). Broadcast if needed:
    e_center = _broadcast_if_1d(e_center, nrec)
    e_upper  = _broadcast_if_1d(e_upper,  nrec)
    e_lower  = _broadcast_if_1d(e_lower,  nrec)

    # Apply sentinel-to-NaN on energy metadata, too:
    e_center[e_center >= 65535.0] = np.nan
    e_upper[e_upper   >= 65535.0] = np.nan
    e_lower[e_lower   >= 65535.0] = np.nan

    # 5) Theta centers/edges (8 anodes). Some files store 1D; broadcast across time.
    t_center = _to_np(data["GROUP_5, DIM2_THETA"])
    t_upper  = _to_np(data["GROUP_6, DIM2_THETA_UPPER"])
    t_lower  = _to_np(data["GROUP_7, DIM2_THETA_LOWER"])

    # Shapes can be (8,) or (nrec, 8). Broadcast to (nrec, 8) if needed:
    t_center = _broadcast_if_1d(t_center, nrec)
    t_upper  = _broadcast_if_1d(t_upper,  nrec)
    t_lower  = _broadcast_if_1d(t_lower,  nrec)

    # Sentinels to NaN for theta metadata too (rare, but safe):
    t_center[t_center >= 65535.0] = np.nan
    t_upper[t_upper   >= 65535.0] = np.nan
    t_lower[t_lower   >= 65535.0] = np.nan

    # 6) Build Dataset
    # Use simple integer index for energy bins (0..62) and store centers/edges as variables.
    ds = xr.Dataset(
        data_vars=dict(
            count_rate=(("time", "energy", "anode"), raw_counts),
            record_dur=(("time",), record_dur),
            energy_center=(("time", "energy"), e_center),
            energy_lower=(("time", "energy"), e_lower),
            energy_upper=(("time", "energy"), e_upper),
            theta_center=(("time", "anode"), t_center),
            theta_lower=(("time", "anode"), t_lower),
            theta_upper=(("time", "anode"), t_upper),
        ),
        coords=dict(
            time=time,
            energy=np.arange(63, dtype=int),
            anode=np.arange(1, 9, dtype=int),
        ),
        attrs=dict(
            source_file=str(xml_path),
            description=(
                "Cassini CAPS/ELS PDS4 → NetCDF with full per-record energy and theta metadata "
                "(centers and edges) and record duration, preserving 63×8 raw anode counts. "
                "Sentinel values (>=65535) are converted to NaN."
            ),
        ),
    )

    # Optional units/attrs
    ds["count_rate"].attrs["units"] = "counts/s"
    ds["record_dur"].attrs["units"] = "s"
    for name in ("energy_center", "energy_lower", "energy_upper"):
        ds[name].attrs["units"] = "eV/q"
    for name in ("theta_center", "theta_lower", "theta_upper"):
        ds[name].attrs["units"] = "deg"

    return ds


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pds4_to_xarray.py <path_to_xml> <out_nc>")
        sys.exit(1)

    xml = sys.argv[1]
    out_nc = sys.argv[2]

    ds = process_file(xml)
    print(ds)
    ds.to_netcdf(out_nc)
    print(f"✓ Wrote {out_nc}")
