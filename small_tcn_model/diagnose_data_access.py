"""
Diagnostic script to check data access issues for training.
"""

from pathlib import Path
import sys

# Get paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
data_dir = project_root / "data" / "true_processed"
labels_dir = project_root / "data" / "zenodo-3946033" / "crossings" / "labels" / "all"

print("=" * 60)
print("Data Access Diagnostic")
print("=" * 60)
print()

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print()

# Check data directory
print("1. Checking data directory...")
print(f"   Path: {data_dir}")
print(f"   Exists: {data_dir.exists()}")
print(f"   Is directory: {data_dir.is_dir() if data_dir.exists() else 'N/A'}")

if data_dir.exists():
    nc_files = list(data_dir.rglob("*.nc"))
    print(f"   NetCDF files found: {len(nc_files)}")
    if nc_files:
        print(f"   Example file: {nc_files[0].name}")
        print(f"   Full path: {nc_files[0]}")
else:
    print("   ERROR: Data directory does not exist!")
print()

# Check labels directory
print("2. Checking labels directory...")
print(f"   Path: {labels_dir}")
print(f"   Exists: {labels_dir.exists()}")
print(f"   Is directory: {labels_dir.is_dir() if labels_dir.exists() else 'N/A'}")

if labels_dir.exists():
    yaml_files = list(labels_dir.glob("*.yaml"))
    print(f"   YAML files found: {len(yaml_files)}")
    if yaml_files:
        print(f"   Example file: {yaml_files[0].name}")
        print(f"   Full path: {yaml_files[0]}")
else:
    print("   ERROR: Labels directory does not exist!")
print()

# Check file matching
print("3. Checking file name matching...")
if data_dir.exists() and labels_dir.exists():
    nc_files = list(data_dir.rglob("*.nc"))
    yaml_files = {f.stem for f in labels_dir.glob("*.yaml")}
    
    matched = 0
    unmatched = []
    
    for nc_file in nc_files[:10]:  # Check first 10
        nc_stem = nc_file.stem
        if nc_stem in yaml_files:
            matched += 1
        else:
            unmatched.append(nc_stem)
    
    print(f"   Checked {min(10, len(nc_files))} NetCDF files")
    print(f"   Matched with YAML: {matched}")
    if unmatched:
        print(f"   Unmatched files (first 5): {unmatched[:5]}")
        # Try to find similar YAML files
        for unmatch in unmatched[:3]:
            similar = [y for y in yaml_files if unmatch[:10] in y]
            if similar:
                print(f"      Similar YAML found: {similar[0]}")
else:
    print("   SKIPPED: Required directories don't exist")
print()

# Test loading a specific file
print("4. Testing data loader import...")
try:
    from data_loader import load_nc_data, get_bow_shock_times, parse_yaml_labels
    print("   ✓ Data loader imports successfully")
    
    if data_dir.exists() and labels_dir.exists():
        nc_files = list(data_dir.rglob("*.nc"))
        if nc_files:
            test_nc = nc_files[0]
            test_yaml = labels_dir / f"{test_nc.stem}.yaml"
            
            print(f"   Testing with: {test_nc.name}")
            print(f"   Expected YAML: {test_yaml.name}")
            print(f"   YAML exists: {test_yaml.exists()}")
            
            if test_yaml.exists():
                try:
                    labels = parse_yaml_labels(test_yaml)
                    if labels:
                        print(f"   ✓ YAML parsed successfully")
                        print(f"   Change points: {len(labels.get('change_points', []))}")
                        if len(labels.get('change_points', [])) > 1:
                            print(f"   ✓ Bow shock event found (index 1)")
                        else:
                            print(f"   ⚠ Warning: Less than 2 change points (no bow shock)")
                    else:
                        print(f"   ✗ YAML parsing returned None")
                except Exception as e:
                    print(f"   ✗ Error parsing YAML: {e}")
            
            try:
                counts, times = load_nc_data(test_nc)
                print(f"   ✓ NetCDF loaded successfully")
                print(f"   Data shape: {counts.shape}")
                print(f"   Time points: {len(times)}")
            except Exception as e:
                print(f"   ✗ Error loading NetCDF: {e}")
                import traceback
                traceback.print_exc()
except Exception as e:
    print(f"   ✗ Error importing data loader: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 60)
print("Diagnostic complete!")
print("=" * 60)
