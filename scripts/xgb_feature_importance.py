import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from joblib import load
import pandas as pd

# === Config ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "xgb_model_win_50.joblib"
FEATURES_FILE = BASE_DIR / "data" / "ml_ready_features.csv"
OUTPUT_DIR = BASE_DIR / "results" / "xgboost"


def main():
    print("\n Generating Feature Importance Plot for XGBoost model...")

    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the trained model
    if not MODEL_PATH.exists():
        print(f" Model not found at {MODEL_PATH}")
        return

    model = load(MODEL_PATH)

    # Load feature names from CSV
    if not FEATURES_FILE.exists():
        print(f" Features file not found at {FEATURES_FILE}")
        return

    df = pd.read_csv(FEATURES_FILE)
    feature_names = df.drop(columns=["ml_label"]).columns.tolist() if "ml_label" in df.columns else df.columns.tolist()

    # Plot and save
    plt.figure(figsize=(10, 6))
    plot_importance(model, max_num_features=20, importance_type='gain', show_values=False)
    plt.title("XGBoost Feature Importance (Top 20)")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(save_path)
    plt.close()

    print(f" Feature importance plot saved to: {save_path}")


if __name__ == "__main__":
    main()
