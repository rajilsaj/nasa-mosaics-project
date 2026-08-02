"""
Entry point for the XGBoost vortex-detection pipeline.
======================================================

Run this from inside the xgboost folder (with your virtual env activated):

    python run.py            -> train the model + evaluate + save results
    python run.py --tune     -> hyperparameter search first, then train
    python run.py --check    -> only verify dependencies and data files, run nothing

First-time setup (from inside the xgboost folder, Windows):

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

# Make sure sibling modules (train_xgb_model.py, tune_xgb_hyperparams.py)
# are importable even if run.py is launched from another directory.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# (pip name, import name)
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
]

REQUIRED_DATA_FILES = [
    os.path.join(DATASETS_DIR, "train_features.csv"),
    os.path.join(DATASETS_DIR, "val_features.csv"),
    os.path.join(DATASETS_DIR, "test_features.csv"),
]

# Needed only for the deployment-realistic sliding-window evaluation;
# training still works without them.
OPTIONAL_DATA_FILES = [
    os.path.join(DATASETS_DIR, "val_sliding_features.csv"),
    os.path.join(DATASETS_DIR, "test_sliding_features.csv"),
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
        print("\nSet up the environment from inside the xgboost folder:")
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

    print("\nOptional files (sliding-window evaluation)...")
    for path in OPTIONAL_DATA_FILES:
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [OK]      {os.path.basename(path)} ({size_mb:.0f} MB)")
        else:
            print(f"  [ABSENT]  {os.path.basename(path)} "
                  f"(sliding evaluation will be skipped)")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Train the XGBoost Mars vortex detector and produce results."
    )
    parser.add_argument("--tune", action="store_true",
                        help="run the hyperparameter search before training")
    parser.add_argument("--check", action="store_true",
                        help="only check dependencies and data files, run nothing")
    args = parser.parse_args()

    print("=" * 70)
    print("XGBOOST PIPELINE RUNNER - MARS VORTEX DETECTION")
    print("=" * 70 + "\n")

    deps_ok = check_dependencies()
    data_ok = check_data_files() if deps_ok else False

    if not (deps_ok and data_ok):
        sys.exit(1)

    if args.check:
        print("\nAll checks passed. Run 'python run.py' to train.")
        return

    total = 4 if args.tune else 3
    step = 1

    param_overrides = None
    if args.tune:
        print("\n" + "=" * 70)
        print(f"STEP {step}/{total}: HYPERPARAMETER SEARCH")
        print("=" * 70 + "\n")
        import tune_xgb_hyperparams
        param_overrides = tune_xgb_hyperparams.main()
        print("\nBest parameters will be applied automatically to training.")
        step += 1

    print("\n" + "=" * 70)
    print(f"STEP {step}/{total}: TRAINING")
    print("=" * 70 + "\n")
    import train_xgb_model
    result = train_xgb_model.main(param_overrides=param_overrides)
    step += 1

    # Deployment-realistic evaluation on the continuous sliding-window
    # stream — the numbers that actually decide RF vs XGBoost.
    print("\n" + "=" * 70)
    print(f"STEP {step}/{total}: SLIDING-WINDOW EVALUATION")
    print("=" * 70)
    import evaluate_sliding_xgb
    evaluate_sliding_xgb.run_evaluation(
        model=result['model'],
        threshold=result['threshold'],
        feature_names=result['feature_names']
    )
    step += 1

    # Model structure PNGs from the freshly trained model. A plotting
    # failure must not discard the training results, so it only warns.
    print("\n" + "=" * 70)
    print(f"STEP {step}/{total}: MODEL STRUCTURE VISUALIZATION")
    print("=" * 70 + "\n")
    import visualize_model
    try:
        visualize_model.generate(model=result['model'])
    except Exception as exc:
        print(f"[WARNING] Visualization failed: {exc}")
        print("The trained model is safe. Retry with: python visualize_model.py")

    print("\nDone. Outputs:")
    print(f"  Models  -> {os.path.join(SCRIPT_DIR, 'models')}")
    print(f"  Results -> {os.path.join(SCRIPT_DIR, 'results')}")
    print("  Structure PNGs: xgb_ensemble_overview.png, xgb_tree_0_structure.png")


if __name__ == "__main__":
    main()
