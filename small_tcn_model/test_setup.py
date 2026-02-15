"""
Diagnostic script to test the training setup and identify issues.
"""

from pathlib import Path
import sys

print("=" * 70)
print("TCN Model Training Setup Diagnostic")
print("=" * 70)
print()

# Get paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
true_data_dir = project_root / "data" / "true_processed"
false_data_dir = project_root / "data" / "false_processed"
labels_dir = project_root / "data" / "zenodo-3946033" / "crossings" / "labels" / "all"

print("1. PATH RESOLUTION")
print("-" * 70)
print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")
print(f"True data dir: {true_data_dir}")
print(f"False data dir: {false_data_dir}")
print(f"Labels dir: {labels_dir}")
print()

print("2. DIRECTORY EXISTENCE")
print("-" * 70)
print(f"True data exists: {true_data_dir.exists()}")
print(f"False data exists: {false_data_dir.exists()}")
print(f"Labels dir exists: {labels_dir.exists()}")
print()

print("3. FILE COUNTS")
print("-" * 70)
if true_data_dir.exists():
    true_files = list(true_data_dir.rglob("*.nc"))
    print(f"True processed NetCDF files: {len(true_files)}")
    if true_files:
        print(f"  Example: {true_files[0].name}")
else:
    print("True processed directory does not exist!")
    print(f"  Expected: {true_data_dir}")

if false_data_dir.exists():
    false_files = list(false_data_dir.rglob("*.nc"))
    print(f"False processed NetCDF files: {len(false_files)}")
    if false_files:
        print(f"  Example: {false_files[0].name}")
else:
    print("False processed directory does not exist!")
    print(f"  Expected: {false_data_dir}")

if labels_dir.exists():
    yaml_files = list(labels_dir.glob("*.yaml"))
    print(f"YAML label files: {len(yaml_files)}")
    if yaml_files:
        print(f"  Example: {yaml_files[0].name}")
else:
    print("Labels directory does not exist!")
    print(f"  Expected: {labels_dir}")
print()

print("4. FILE MATCHING TEST")
print("-" * 70)
if true_data_dir.exists() and labels_dir.exists():
    true_files = list(true_data_dir.rglob("*.nc"))
    yaml_files = {f.stem for f in labels_dir.glob("*.yaml")}
    
    matched = 0
    unmatched = []
    for nc_file in true_files[:10]:
        if nc_file.stem in yaml_files:
            matched += 1
        else:
            unmatched.append(nc_file.stem)
    
    print(f"Checked {min(10, len(true_files))} true_processed files")
    print(f"Matched with YAML: {matched}/{min(10, len(true_files))}")
    if unmatched:
        print(f"Unmatched files (first 3): {unmatched[:3]}")
        # Check for similar names
        for unmatch in unmatched[:2]:
            similar = [y for y in yaml_files if unmatch[:10] in y or y[:10] in unmatch]
            if similar:
                print(f"  '{unmatch}' - similar YAML found: {similar[0]}")
else:
    print("Cannot check matching - directories missing")
print()

print("5. IMPORT TESTS")
print("-" * 70)
try:
    import torch
    print(f"✓ PyTorch imported: version {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    import numpy as np
    print(f"✓ NumPy imported: version {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas imported: version {pd.__version__}")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

try:
    import xarray as xr
    print(f"✓ XArray imported: version {xr.__version__}")
except ImportError as e:
    print(f"✗ XArray import failed: {e}")

try:
    from sklearn.metrics import accuracy_score
    print(f"✓ Scikit-learn imported")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")
print()

print("6. DATA LOADER IMPORT")
print("-" * 70)
try:
    sys.path.insert(0, str(script_dir))
    from data_loader import create_data_loaders, load_nc_data
    print("✓ Data loader imported successfully")
except ImportError as e:
    print(f"✗ Data loader import failed: {e}")
    import traceback
    traceback.print_exc()
print()

print("7. TEST DATA LOADING")
print("-" * 70)
if true_data_dir.exists():
    true_files = list(true_data_dir.rglob("*.nc"))
    if true_files:
        test_file = true_files[0]
        print(f"Testing with: {test_file.name}")
        try:
            sys.path.insert(0, str(script_dir))
            from data_loader import load_nc_data
            counts, times = load_nc_data(test_file)
            print(f"✓ NetCDF loaded successfully")
            print(f"  Data shape: {counts.shape}")
            print(f"  Time points: {len(times)}")
        except Exception as e:
            print(f"✗ Failed to load NetCDF: {e}")
            import traceback
            traceback.print_exc()
else:
    print("Cannot test - true_processed directory missing")
print()

print("8. TEST DATA LOADER CREATION")
print("-" * 70)
if (true_data_dir.exists() and false_data_dir.exists() and labels_dir.exists()):
    try:
        sys.path.insert(0, str(script_dir))
        from data_loader import create_data_loaders
        
        print("Attempting to create data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            true_processed_dir=true_data_dir,
            false_processed_dir=false_data_dir,
            labels_dir=labels_dir,
            sequence_length=100,
            stride=10,
            batch_size=2,  # Small batch for testing
            normalize=True,
            window_size=5,
        )
        print("✓ Data loaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")
        
        # Try to get a batch
        try:
            sample_seq, sample_labels = next(iter(train_loader))
            print(f"  Sample batch shape: {sample_seq.shape}")
            print(f"  Sample labels shape: {sample_labels.shape}")
            print(f"  Positive labels in sample: {(sample_labels == 1).sum().item()}")
            print(f"  Negative labels in sample: {(sample_labels == 0).sum().item()}")
        except Exception as e:
            print(f"✗ Failed to get batch: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"✗ Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Cannot test - required directories missing")
print()

print("=" * 70)
print("Diagnostic complete!")
print("=" * 70)
