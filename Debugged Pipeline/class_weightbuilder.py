import os

import joblib
import numpy as np

# ===== PATHS =====
DATA_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\windows_2004\scaled"
TRAIN_Y = os.path.join(DATA_DIR, "train_y.npy")
SAVE_PATH = os.path.join(DATA_DIR, "class_weights.pkl")
# =================


def main() -> None:
    y = np.load(TRAIN_Y)
    if len(y) == 0:
        raise RuntimeError("train_y.npy is empty.")

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)

    class_weights = {}
    for cls, count in zip(classes.tolist(), counts.tolist()):
        class_weights[int(cls)] = float(total / (n_classes * count))

    joblib.dump(class_weights, SAVE_PATH)

    print("Train samples:", total)
    print("Class counts:", dict(zip(classes.tolist(), counts.tolist())))
    print("Class weights:", class_weights)
    print("Saved class weights to:", SAVE_PATH)


if __name__ == "__main__":
    main()