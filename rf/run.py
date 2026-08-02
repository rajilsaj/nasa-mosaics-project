"""
Entry point for the Random Forest vortex-detection pipeline.
============================================================

Run this from inside the rf folder (with your virtual env activated):

    python run.py            -> train the model + evaluate + save results
    python run.py --check    -> only verify dependencies and data files, run nothing

First-time setup (from inside the rf folder, Windows):

    python -m venv env
    .\\env\\Scripts\\activate
    pip install -r requirements.txt
    python run.py
"""

import argparse
import importlib
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Make sure the sibling module (train_rf_model.py) is importable even if
# run.py is launched from another directory.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# (pip name, import name)
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
]

REQUIRED_DATA_FILES = [
    os.path.join(DATASETS_DIR, "train_features.csv"),
    os.path.join(DATASETS_DIR, "val_features.csv"),
    os.path.join(DATASETS_DIR, "test_features.csv"),
]


def check_dependencies():
    """Verify every required package is importable. Returns True if OK."""
    print("Checking dependencies...")
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK]      {pip_name} ({version})")
        except ImportError:
            print(f"  [MISSING] {pip_name}")
            missing.append(pip_name)

    if missing:
        print("\nMissing packages: " + ", ".join(missing))
        print("\nSet up the environment from inside the rf folder:")
        print("  python -m venv env")
        print("  .\\env\\Scripts\\activate")
        print("  pip install -r requirements.txt")
        return False
    return True


def check_data_files():
    """Verify the three dataset splits exist in the datasets folder."""
    print("\nChecking data files (datasets folder)...")
    ok = True
    for path in REQUIRED_DATA_FILES:
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  [OK]      {os.path.basename(path)} ({size_kb:.0f} KB)")
        else:
            print(f"  [MISSING] {path}")
            ok = False

    if not ok:
        print("\nOne or more dataset files are missing from the datasets folder.")
        print("Expected in datasets/: train_features.csv, val_features.csv, "
              "test_features.csv")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Train the Random Forest Mars vortex detector and produce results."
    )
    parser.add_argument("--check", action="store_true",
                        help="only check dependencies and data files, run nothing")
    args = parser.parse_args()

    print("=" * 70)
    print("RANDOM FOREST PIPELINE RUNNER - MARS VORTEX DETECTION")
    print("=" * 70 + "\n")

    deps_ok = check_dependencies()
    data_ok = check_data_files() if deps_ok else False

    if not (deps_ok and data_ok):
        sys.exit(1)

    if args.check:
        print("\nAll checks passed. Run 'python run.py' to train.")
        return

    print("\n" + "=" * 70)
    print("STEP 1/2: TRAINING")
    print("=" * 70 + "\n")
    import train_rf_model
    train_rf_model.main()

    # Model structure PNGs, generated automatically from the model that
    # was just saved. A plotting failure must not discard the training
    # results, so it only warns.
    print("\n" + "=" * 70)
    print("STEP 2/2: MODEL STRUCTURE VISUALIZATION")
    print("=" * 70 + "\n")
    import visualize_model
    try:
        visualize_model.generate()
    except Exception as exc:
        print(f"[WARNING] Visualization failed: {exc}")
        print("The trained model is safe. Retry with: python visualize_model.py")

    print("\nDone. Outputs:")
    print(f"  Models  -> {os.path.join(SCRIPT_DIR, 'models')}")
    print(f"  Results -> {os.path.join(SCRIPT_DIR, 'results')}")
    print("  Structure PNGs: rf_forest_overview.png, rf_tree_0_structure.png")


if __name__ == "__main__":
    main()
