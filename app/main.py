import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"

# === Define the script execution order ===
SCRIPT_SEQUENCE = [
    ("address.py", "Generating labeled training windows and computing thresholds..."),
    ("xgboost_vortex_detector.py", "Training and tuning XGBoost classifier..."),
    ("modeladdress.py", "Training and tuning Random Forest classifier..."),
    ("ts_vortex_detector_weighted.py", "Running time-series weighted vortex detection..."),
    ("vortex_prediction_model.py", "Running unified vortex model with configurable classifier..."),
]


def run_script(script_name, message):
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f" {script_name} not found at {script_path}")
        return

    print(f"\n {message}")
    result = subprocess.run(["python", str(script_path)], capture_output=True, text=True)

    if result.returncode == 0:
        print(f" {script_name} completed successfully.")
    else:
        print(f" Error in {script_name}:")
        print(result.stderr)


def main():
    print("==== Vortex Detection Project Runner ====")
    for script, message in SCRIPT_SEQUENCE:
        run_script(script, message)


if __name__ == "__main__":
    main()
