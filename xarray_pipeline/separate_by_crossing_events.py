#!/usr/bin/env python3
"""
Separate .nc files from NEW_PROCESSED_DIR into two folders:
- true_processed: Files with at least one crossing event (crossing_event = 1)
- false_processed: Files with no crossing events (all crossing_event = 0)
"""

import xarray as xr
import numpy as np
from pathlib import Path
import shutil


def has_crossing_events(nc_path):
    """
    Check if a .nc file has any crossing events.
    
    Args:
        nc_path: Path to .nc file
    
    Returns:
        True if any crossing_event values are 1, False otherwise
    """
    try:
        ds = xr.open_dataset(nc_path)
        
        if 'crossing_event' not in ds:
            print(f"  Warning: No crossing_event variable found")
            ds.close()
            return False
        
        crossing_events = ds['crossing_event'].values
        has_events = np.any(crossing_events == 1)
        n_events = np.sum(crossing_events == 1)
        
        ds.close()
        
        return has_events, n_events
        
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False, 0


def main():
    # Define paths
    NEW_PROCESSED_DIR = Path("/Users/jacobhuss/Cassini/data/new_processed")
    TRUE_PROCESSED_DIR = Path("/Users/jacobhuss/Cassini/data/true_processed")
    FALSE_PROCESSED_DIR = Path("/Users/jacobhuss/Cassini/data/false_processed")
    
    # Find all .nc files
    nc_files = sorted(NEW_PROCESSED_DIR.rglob("*.nc"))
    
    print("=" * 80)
    print("SEPARATING FILES BY CROSSING EVENTS")
    print("=" * 80)
    print(f"Source directory: {NEW_PROCESSED_DIR}")
    print(f"Target directory (with events): {TRUE_PROCESSED_DIR}")
    print(f"Target directory (no events): {FALSE_PROCESSED_DIR}")
    print()
    print(f"Found {len(nc_files)} .nc files to process")
    print("=" * 80)
    print()
    
    # Counters
    files_with_events = 0
    files_without_events = 0
    total_events = 0
    errors = 0
    
    # Process each file
    for i, nc_path in enumerate(nc_files, 1):
        print(f"[{i}/{len(nc_files)}] {nc_path.name}")
        
        try:
            # Check for crossing events
            has_events, n_events = has_crossing_events(nc_path)
            
            # Determine output path (maintain folder structure)
            rel_path = nc_path.relative_to(NEW_PROCESSED_DIR)
            
            if has_events:
                # Copy to true_processed
                output_path = TRUE_PROCESSED_DIR / rel_path
                files_with_events += 1
                total_events += n_events
                print(f"  -> TRUE_PROCESSED ({n_events} crossing events)")
            else:
                # Copy to false_processed
                output_path = FALSE_PROCESSED_DIR / rel_path
                files_without_events += 1
                print(f"  -> FALSE_PROCESSED (no crossing events)")
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(nc_path, output_path)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            continue
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(nc_files)}")
    print(f"Files WITH crossing events: {files_with_events} (copied to {TRUE_PROCESSED_DIR})")
    print(f"Files WITHOUT crossing events: {files_without_events} (copied to {FALSE_PROCESSED_DIR})")
    print(f"Total crossing events found: {total_events}")
    print(f"Errors: {errors}")
    print("=" * 80)
    
    # Show breakdown by folder
    if files_with_events > 0:
        print()
        print("Files with crossing events by month:")
        for month_dir in sorted(TRUE_PROCESSED_DIR.rglob("*/")):
            if month_dir.is_dir():
                month_files = list(month_dir.glob("*.nc"))
                if month_files:
                    print(f"  {month_dir.name}: {len(month_files)} files")


if __name__ == '__main__':
    main()

