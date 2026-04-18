import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ===== PATHS =====
IN_DIR  = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/windows_2005_focus6h"
OUT_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/windows_2005_focus6h/scaled"
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
# =================

def load_split(split: str):
    X = np.load(os.path.join(IN_DIR, f"{split}_X.npy"))
    y = np.load(os.path.join(IN_DIR, f"{split}_y.npy"))
    return X, y

def scale_with(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X.astype(np.float32)
    n, w, f = X.shape
    X2 = scaler.transform(X.reshape(-1, f))
    return X2.reshape(n, w, f).astype(np.float32)

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    X_train, _ = load_split("train")
    if len(X_train) == 0:
        raise RuntimeError("Training windows are empty. Build windows first.")
    _, _, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, f))
    joblib.dump(scaler, SCALER_PATH)
    print("Saved scaler to:", SCALER_PATH)

    for split in ("train", "val", "test"):
        X, y = load_split(split)
        Xs = scale_with(scaler, X)
        np.save(os.path.join(OUT_DIR, f"{split}_X.npy"), Xs)
        np.save(os.path.join(OUT_DIR, f"{split}_y.npy"), y.astype(np.uint8))

        # compute class weights and save for training script
        if split == "train":
            counts = np.bincount(y, minlength=3)
            total = len(y)
            weights = {i: total / (3 * counts[i]) if counts[i] > 0 else 1.0 for i in range(3)}
            weights_path = os.path.join(OUT_DIR, "class_weights.pkl")
            joblib.dump(weights, weights_path)
            print("Class weights:", weights)
            print("Saved class weights to:", weights_path)

        print(
            f"{split.upper()}: X {Xs.shape}, y {y.shape}, "
            f"positives {int((y > 0).sum())}, "
            f"positive_frac {float((y > 0).mean()) if len(y) else 0.0:.4f}"
        )
    print("Done.")

if __name__ == "__main__":
    main()
