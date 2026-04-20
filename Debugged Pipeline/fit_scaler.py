import os

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ===== PATHS =====
IN_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\windows_2004"
OUT_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\dataset_index\windows_2004\scaled"
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")
# =================


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(os.path.join(IN_DIR, f"{split}_X.npy"), mmap_mode="r")
    y = np.load(os.path.join(IN_DIR, f"{split}_y.npy"), mmap_mode="r")
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
        print(
            f"{split.upper()}: X {Xs.shape}, y {y.shape}, "
            f"positives {int((y > 0).sum())}, positive_frac {float((y > 0).mean()) if len(y) else 0.0}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
