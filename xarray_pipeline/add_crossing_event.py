#!/usr/bin/env python3
"""
Add 'crossing_event' variable to all .nc files in processed folder.
Outputs modified files to new_processed folder with same directory structure.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime

def parse_yaml_labels(yaml_path):
    """
    Parse crossing event labels from YAML file (manual parsing - no yaml library needed).
    Returns dict with change_points, bimodality, negative_ions
    """
    if not yaml_path.exists():
        return None
    
    # Parse YAML manually to avoid yaml library dependency
    data = {
        'change_points': [],
        'bimodality': [],
        'negative_ions': []
    }
    
    with open(yaml_path, 'r') as f:
        current_key = None
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if line is a key
            if line.endswith(':'):
                current_key = line[:-1].strip()
                # Handle "key: []" format
                if current_key not in data:
                    data[current_key] = []
            elif line.startswith('- '):
                # This is a list item
                value = line[2:].strip()
                if current_key and current_key in data:
                    data[current_key].append(value)
    
    return data


def create_crossing_event_variable(ds, yaml_labels=None):
    """
    Create crossing_event variable for the dataset.
    
    Creates a 1D array with one value per time point (row):
    - Default: All values are 0 (False - no crossing event)
    - If YAML labels exist: Set to 1 (True) ONLY when UTC time matches a change_point
    
    Args:
        ds: xarray Dataset with 'time' coordinate (UTC timestamps)
        yaml_labels: Optional dict from YAML file containing 'change_points' list
                     (format: 'DD-MM-YYYY/HH:MM:SS')
    
    Returns: xr.DataArray with shape (time,) containing 0s and 1s
    """
    n_times = len(ds['time'])
    # Initialize ALL time points to 0 (False - no crossing event)
    # This creates one value per row (time point) in the dataset
    crossing_events = np.zeros(n_times, dtype=np.int8)
    
    # Set to 1 (True) ONLY if UTC time in dataset matches a UTC time in YAML labels
    if yaml_labels and 'change_points' in yaml_labels:
        change_points = yaml_labels.get('change_points', [])
        
        # For each change_point UTC time in YAML, find matching row in dataset
        for cp_str in change_points:
            if cp_str:
                try:
                    # Parse YAML UTC time format: DD-MM-YYYY/HH:MM:SS
                    cp_dt = datetime.strptime(cp_str, '%d-%m-%Y/%H:%M:%S')
                    
                    # Find the dataset row (time index) with UTC closest to this YAML UTC
                    time_array = ds['time'].values  # UTC timestamps from dataset
                    time_diffs = np.abs(time_array - np.datetime64(cp_dt))
                    closest_idx = np.argmin(time_diffs)
                    
                    # Mark this row as having a crossing event (set to 1 = True)
                    crossing_events[closest_idx] = 1
                    print(f"    UTC match found: Row {closest_idx} - {cp_str} -> crossing_event=True")
                except Exception as e:
                    print(f"  Warning: Could not parse UTC time '{cp_str}': {e}")
    
    # Create DataArray with metadata
    crossing_var = xr.DataArray(
        data=crossing_events,
        dims=('time',),
        coords={'time': ds['time']},
        attrs={
            'long_name': 'Crossing Event Flag',
            'description': 'Binary flag indicating magnetosphere boundary crossing events',
            'units': '1',
            'flag_values': '0, 1',
            'flag_meanings': 'no_event crossing_event',
            'comment': 'Derived from manual labels of change points'
        }
    )
    
    return crossing_var


def process_nc_file(input_path, output_path, labels_dir=None):
    """
    Process a single .nc file: add crossing_event variable and save.
    
    Args:
        input_path: Path to input .nc file
        output_path: Path to output .nc file
        labels_dir: Optional path to directory containing YAML label files
    """
    # Load dataset
    ds = xr.open_dataset(input_path)
    
    # Fix missing data flags (65535) in count_rate variable
    # This is the standard "no data" marker from the original PDS4 files
    if 'count_rate' in ds:
        missing_value = 65535.0
        count_rate_data = ds['count_rate'].values
        n_missing = np.sum(count_rate_data == missing_value)
        if n_missing > 0:
            print(f"  Fixing {n_missing} missing data flags (65535 -> NaN) in count_rate")
            count_rate_data[count_rate_data == missing_value] = np.nan
            # Update the dataset with cleaned data
            ds['count_rate'].values = count_rate_data
    
    # Check if crossing_event already exists
    if 'crossing_event' in ds:
        print(f"  Warning: 'crossing_event' already exists in {input_path.name}, skipping...")
        ds.close()
        return
    
    # Try to load YAML labels if labels_dir provided
    yaml_labels = None
    if labels_dir:
        yaml_path = labels_dir / f"{input_path.stem}.yaml"
        yaml_labels = parse_yaml_labels(yaml_path)
        if yaml_labels:
            print(f"  Found labels for {input_path.name}")
    
    # Create crossing_event variable
    crossing_var = create_crossing_event_variable(ds, yaml_labels)
    
    # Add to dataset
    ds['crossing_event'] = crossing_var
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Encoding for compression
    encoding = {
        'crossing_event': {
            'zlib': True,
            'complevel': 4,
            'dtype': 'int8'
        }
    }
    
    # Save to new file
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"  Wrote: {output_path}")
    
    ds.close()


def main():
    # Define paths
    processed_dir = Path("/Users/jacobhuss/Cassini/data/processed")
    new_processed_dir = Path("/Users/jacobhuss/Cassini/data/new_processed")
    labels_dir = Path("/Users/jacobhuss/Cassini/crossings/labels/all")
    
    # Optional: set to None if you don't want to use YAML labels
    # If None, all crossing_event values will be 0 (False)
    # If labels_dir provided, crossing_event is 0 by default and set to 1 ONLY at yaml change_point times
    # labels_dir = None
    
    # Find all .nc files
    nc_files = sorted(processed_dir.rglob("*.nc"))
    
    print("=" * 80)
    print("ADDING CROSSING_EVENT VARIABLE TO .NC FILES")
    print("=" * 80)
    print(f"Input directory:  {processed_dir}")
    print(f"Output directory: {new_processed_dir}")
    print(f"Labels directory: {labels_dir if labels_dir else 'None'}")
    print()
    print(f"Found {len(nc_files)} .nc files to process")
    print()
    print("BEHAVIOR:")
    print("  - All crossing_event values default to 0 (False - no crossing event)")
    print("  - If YAML label exists with change_points, set to 1 (True) ONLY at those times")
    print("  - If no YAML label exists, all values remain 0")
    print("=" * 80)
    print()
    
    files_with_labels = 0
    files_without_labels = 0
    
    # Process each file
    for i, nc_path in enumerate(nc_files, 1):
        # Create relative path to maintain directory structure
        rel_path = nc_path.relative_to(processed_dir)
        output_path = new_processed_dir / rel_path
        
        print(f"[{i}/{len(nc_files)}] Processing: {nc_path.name}")
        
        try:
            # Check if labels exist before processing
            has_labels = False
            if labels_dir:
                yaml_path = labels_dir / f"{nc_path.stem}.yaml"
                if yaml_path.exists():
                    has_labels = True
                    files_with_labels += 1
                else:
                    files_without_labels += 1
                    print(f"  No YAML labels found - all crossing_event will be 0")
            
            process_nc_file(nc_path, output_path, labels_dir)
        except Exception as e:
            print(f"  ERROR processing {nc_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(nc_files)}")
    print(f"Files with YAML labels: {files_with_labels}")
    print(f"Files without labels (all 0s): {files_without_labels}")
    print(f"Output saved to: {new_processed_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()

