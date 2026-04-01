"""Plot monthly spectrograms for processed files flagged 'True'."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm


ROOT = Path("/Users/jacobhuss/Cassini")
TRUE_PROCESSED_DIR = ROOT / "data/true_processed"
LABELS_DIR = ROOT / "crossings/labels/all"


def parse_yaml_labels(yaml_path: Path) -> dict[str, list[str]] | None:
    if not yaml_path.exists():
        return None

    data: dict[str, list[str]] = {
        "change_points": [],
        "bimodality": [],
        "negative_ions": [],
    }

    with open(yaml_path, "r", encoding="utf-8") as fh:
        current_key: str | None = None
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":"):
                current_key = line[:-1].strip()
                if current_key not in data:
                    data[current_key] = []
            elif line.startswith("- "):
                value = line[2:].strip()
                if current_key and current_key in data:
                    data[current_key].append(value)
    return data


def add_crossing_markers(ax: plt.Axes, labels: dict[str, list[str]]) -> list[tuple[float, str, str]]:
    change_points = labels.get("change_points") if labels else None
    if not change_points:
        return []

    markers: list[tuple[float, str, str]] = []
    for idx, cp in enumerate(change_points):
        try:
            dt = datetime.strptime(cp, "%d-%m-%Y/%H:%M:%S")
        except ValueError:
            try:
                dt = datetime.strptime(cp, "%d-%m-%Y/%H:%M")
            except ValueError:
                continue
        x_val = mdates.date2num(dt)
        if idx == 0:
            color = "red"
            label = "Magnetopause"
        elif idx == 1:
            color = "black"
            label = "Bow Shock"
        else:
            color = "blue"
            label = f"Event {idx + 1}"
        ax.axvline(x_val, color=color, linewidth=2, alpha=0.8)
        markers.append((x_val, color, label))
    return markers
def find_labels_for_month(files: Iterable[Path]) -> dict[Path, dict[str, list[str]]]:
    labels: dict[Path, dict[str, list[str]]] = {}
    for nc_path in files:
        label_path = LABELS_DIR / f"{nc_path.stem}.yaml"
        labels[nc_path] = parse_yaml_labels(label_path)
    return labels


def extract_energy(ds: xr.Dataset) -> dict[str, np.ndarray]:
    edge_vars = {"energy_lower", "energy_upper"}
    if edge_vars <= set(ds.variables):
        lower = ds["energy_lower"].values.astype(float)
        upper = ds["energy_upper"].values.astype(float)
        if "energy_center" in ds.variables:
            center = ds["energy_center"].values.astype(float)
        else:
            center = (lower + upper) / 2.0
        return {"center": center, "lower": lower, "upper": upper}
    energy = ds["energy"].values.astype(float)
    return {"center": energy, "lower": energy, "upper": energy}


def fill_nan_edges(edges: np.ndarray) -> np.ndarray:
    filled = edges.copy()
    rows, cols = filled.shape

    for col in range(cols):
        column = filled[:, col]
        finite = np.isfinite(column)
        if finite.all():
            continue
        if finite.sum() >= 2:
            x = np.arange(rows)
            column[~finite] = np.interp(x[~finite], x[finite], column[finite])
        elif finite.sum() == 1:
            column[:] = column[finite][0]
        filled[:, col] = column

    finite_cols = np.where(np.isfinite(filled).any(axis=0))[0]
    if finite_cols.size == 0:
        raise ValueError("Energy edges contain no finite values.")

    for col in range(cols):
        if not np.isfinite(filled[:, col]).any():
            nearest = finite_cols[np.argmin(np.abs(finite_cols - col))]
            filled[:, col] = filled[:, nearest]

    filled = np.where(np.isfinite(filled), filled, 1.0)
    return filled


def compute_energy_edges(energies: dict[str, np.ndarray]) -> np.ndarray:
    center = energies["center"]
    lower = energies["lower"]
    upper = energies["upper"]

    if center.ndim == 1:
        if center.size < 2:
            base = center[0]
            edges = np.array([base * 0.8, base * 1.2], dtype=float)
        elif not np.array_equal(lower, center) or not np.array_equal(upper, center):
            edges = np.empty(center.size + 1, dtype=float)
            edges[:-1] = lower
            edges[-1] = upper[-1]
        else:
            log_e = np.log10(center)
            log_edges = np.concatenate(
                [
                    [log_e[0] - (log_e[1] - log_e[0]) / 2.0],
                    (log_e[:-1] + log_e[1:]) / 2.0,
                    [log_e[-1] + (log_e[-1] - log_e[-2]) / 2.0],
                ]
            )
            edges = 10 ** log_edges

        positive = edges[edges > 0]
        if positive.size:
            min_positive = positive.min()
            edges = np.where(edges > 0, edges, min_positive)
        else:
            edges = np.ones_like(edges)
        return edges

    if center.ndim != 2:
        raise ValueError(f"Unexpected energy array shape: {center.shape}")

    n_time, n_energy = center.shape
    edges = np.empty((n_energy + 1, n_time), dtype=float)
    edges[:-1, :] = lower.T
    edges[-1, :] = upper[:, -1]

    edges = fill_nan_edges(edges)
    edges = np.column_stack([edges, edges[:, -1]])

    positive = edges[edges > 0]
    if positive.size == 0:
        edges[:] = 1.0
    else:
        min_positive = positive.min()
        edges = np.where(edges > 0, edges, min_positive)

    return edges


def average_anodes(counts: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(counts, axis=2)


def interpolate_nan_columns(data: np.ndarray) -> np.ndarray:
    result = data.copy()
    for col_idx in range(result.shape[1]):
        column = result[:, col_idx]
        finite_mask = np.isfinite(column)
        if finite_mask.sum() < 2:
            continue
        if not np.all(finite_mask):
            missing_idx = np.where(~finite_mask)[0]
            known_idx = np.where(finite_mask)[0]
            known_vals = column[finite_mask]
            column[~finite_mask] = np.interp(missing_idx, known_idx, known_vals)
            result[:, col_idx] = column
    return result


def compute_time_edges(times: np.ndarray) -> np.ndarray:
    numeric = mdates.date2num(pd.to_datetime(times).to_pydatetime())
    if numeric.size < 2:
        delta = 1 / (24 * 60)
        return np.array([numeric[0] - delta, numeric[0] + delta])
    diffs = np.diff(numeric)
    return np.concatenate(
        [
            [numeric[0] - diffs[0] / 2.0],
            (numeric[:-1] + numeric[1:]) / 2.0,
            [numeric[-1] + diffs[-1] / 2.0],
        ]
    )


def load_nc(nc_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    ds = xr.open_dataset(nc_path)
    counts = ds["count_rate"].values.astype(float)
    counts[counts == 65535.0] = np.nan
    energy = extract_energy(ds)
    times = ds["time"].values
    ds.close()
    return counts, energy, times


def ensure_energy_time_alignment(energy: dict[str, np.ndarray], time_len: int) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for key in ("center", "lower", "upper"):
        arr = energy[key]
        if arr.ndim == 1:
            aligned[key] = np.tile(arr, (time_len, 1))
        elif arr.ndim == 2:
            if arr.shape[0] != time_len:
                raise ValueError(f"Energy array for '{key}' has shape {arr.shape} but expected time dimension {time_len}")
            aligned[key] = arr
        else:
            raise ValueError(f"Unexpected ndim for energy '{key}': {arr.ndim}")
    return aligned


def plot_nc(ax, counts: np.ndarray, energy_info: dict[str, np.ndarray], times: np.ndarray):
    counts = average_anodes(counts)
    counts = interpolate_nan_columns(counts)

    finite_positive = counts[np.isfinite(counts) & (counts > 0)]
    if finite_positive.size:
        vmin = max(1.0, finite_positive.min())
        vmax = finite_positive.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    time_edges = compute_time_edges(times)
    energy_edges = compute_energy_edges(energy_info)

    if energy_edges.ndim == 1:
        mesh = ax.pcolormesh(
            time_edges,
            energy_edges,
            counts.T,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )
        y_min, y_max = energy_edges.min(), energy_edges.max()
    else:
        time_grid = np.tile(time_edges, (energy_edges.shape[0], 1))
        mesh = ax.pcolormesh(
            time_grid,
            energy_edges,
            counts.T,
            shading="auto",
            cmap="viridis",
            norm=norm,
        )
        y_min = np.nanmin(energy_edges)
        y_max = np.nanmax(energy_edges)

    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Energy (eV/q)")
    ax.set_yscale("log")
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(max(1.0, y_min), min(1e4, y_max))
    else:
        ax.set_ylim(1.0, 1e4)
    ax.set_yticks([1, 10, 100, 1000, 10000])
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}"))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y/%H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Counts / s (interpolated anode mean)")
    return cbar


def collect_nc_files(root: Path) -> dict[tuple[int, int], list[Path]]:
    by_month: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in root.rglob("*.nc"):
        if not path.is_file():
            continue
        try:
            with xr.open_dataset(path) as ds:
                times = pd.to_datetime(ds["time"].values)
            if times.size == 0:
                continue
            start = times[0]
        except Exception:
            continue
        key = (start.year, start.month)
        by_month[key].append(path)
    return by_month


def plot_month(
    month_key: tuple[int, int],
    files: list[Path],
    output_dir: Path | None,
    show: bool,
    show_crossings: bool,
) -> None:
    year, month = month_key
    files = sorted(files)

    counts_list: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    times_list: list[np.ndarray] = []
    energy_bins: set[int] = set()

    for nc_path in files:
        counts, energy_info, times = load_nc(nc_path)
        time_len, energy_len, _ = counts.shape
        energy_bins.add(energy_len)
        aligned_energy = ensure_energy_time_alignment(energy_info, time_len)

        counts_list.append(counts)
        centers.append(aligned_energy["center"])
        lowers.append(aligned_energy["lower"])
        uppers.append(aligned_energy["upper"])
        times_list.append(times)

    if len(energy_bins) > 1:
        raise ValueError(f"Energy dimension mismatch within {year}-{month:02d}: {energy_bins}")

    combined_counts = np.concatenate(counts_list, axis=0)
    combined_center = np.concatenate(centers, axis=0)
    combined_lower = np.concatenate(lowers, axis=0)
    combined_upper = np.concatenate(uppers, axis=0)
    combined_times = np.concatenate(times_list)

    combined_energy = {"center": combined_center, "lower": combined_lower, "upper": combined_upper}

    fig, ax = plt.subplots(figsize=(12, 6))
    cbar = plot_nc(ax, combined_counts, combined_energy, combined_times)

    month_name = datetime(year, month, 1).strftime("%B %Y")
    ax.set_title(f"TRUE_PROCESSED Monthly Spectrogram - {month_name}")

    legend_entries: list[tuple[str, str]] = []
    if show_crossings:
        month_labels = find_labels_for_month(files)
        for nc_path in files:
            labels = month_labels.get(nc_path)
            if labels:
                markers = add_crossing_markers(ax, labels)
                for _, color, label in markers:
                    legend_entries.append((color, label))

    if legend_entries:
        unique_entries: dict[str, str] = {}
        for color, label in legend_entries:
            unique_entries[label] = color
        handles = [
            plt.Line2D([0], [0], color=color, linewidth=2, label=label)
            for label, color in unique_entries.items()
        ]

        # Place legend to the right of the colorbar
        cbar_position = cbar.ax.get_position()
        legend_x = cbar_position.x1 + 0.3
        legend_y = cbar_position.y1
        legend_height = cbar_position.height

        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(legend_x, legend_y),
            borderaxespad=0,
            frameon=True,
        )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / f"true_{year}{month:02d}.png"
        plt.savefig(outfile, bbox_inches="tight", dpi=150)
        print(f"Saved {outfile}")
        plt.close(fig)
    else:
        if show:
            plt.show()
        else:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TRUE_PROCESSED_DIR monthly spectrograms.")
    parser.add_argument(
        "--true-dir",
        dest="true_dir",
        default=str(TRUE_PROCESSED_DIR),
        help="Directory containing true processed .nc files.",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Optional directory to save plots instead of displaying.",
    )
    parser.add_argument(
        "--month",
        dest="month",
        default=None,
        help="Optional month filter in YYYY-MM format.",
    )
    parser.add_argument(
        "--crossings",
        dest="show_crossings",
        action="store_true",
        help="Overlay crossing events (Magnetopause/Bow Shock) when labels exist.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not display plots interactively.",
    )
    args = parser.parse_args()

    true_dir = Path(args.true_dir).expanduser().resolve()
    if not true_dir.exists():
        raise FileNotFoundError(f"True processed directory not found: {true_dir}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

    files_by_month = collect_nc_files(true_dir)
    if not files_by_month:
        print(f"No .nc files found under {true_dir}")
        return

    if args.month:
        try:
            month_dt = datetime.strptime(args.month, "%Y-%m")
            key = (month_dt.year, month_dt.month)
        except ValueError:
            raise ValueError("Month must be in YYYY-MM format.")
        month_files = files_by_month.get(key, [])
        if not month_files:
            print(f"No files found for {args.month}")
            return
        plot_month(key, month_files, output_dir, args.show, args.show_crossings)
    else:
        for key, files in sorted(files_by_month.items()):
            print(f"Plotting {len(files)} files for {key[0]}-{key[1]:02d}")
            plot_month(key, files, output_dir, args.show, args.show_crossings)


if __name__ == "__main__":
    main()

