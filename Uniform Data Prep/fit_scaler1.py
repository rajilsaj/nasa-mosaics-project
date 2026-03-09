import os
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler

IN_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\data\dataset_index\windows_2004"
OUT_DIR = r"C:\Users\PC\Documents\GitHub\nasa-mosaics-project\data\dataset_index\windows_2004\scaled"
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")

os.makedirs(OUT_DIR, exist_ok=True)

def load_split(split):
    X = np.load(os.path.join(IN_DIR, f"{split}_X.npy"))  # (N, W, 63)
    y = np.load(os.path.join(IN_DIR, f"{split}_y.npy"))  # (N,)
    return X, y

def scale_with(scaler, X):
    # reshape to 2D: (N*W, 63) -> scale -> back to (N, W, 63)
    N, W, F = X.shape
    X2 = X.reshape(-1, F)
    X2 = scaler.transform(X2)
    return X2.reshape(N, W, F).astype(np.float32)

# ---- fit scaler on TRAIN only ----
X_train, y_train = load_split("train")
N, W, F = X_train.shape

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, F))

joblib.dump(scaler, SCALER_PATH)
print("✅ Saved scaler to:", SCALER_PATH)

# ---- transform and save all splits ----
for split in ["train", "val", "test"]:
    X, y = load_split(split)
    Xs = scale_with(scaler, X)

    np.save(os.path.join(OUT_DIR, f"{split}_X.npy"), Xs)
    np.save(os.path.join(OUT_DIR, f"{split}_y.npy"), y.astype(np.uint8))

    print(f"{split.upper()}: X {Xs.shape}, y {y.shape}, pos {int(y.sum())}, pos_frac {float(y.mean()) if len(y) else 0.0}")

print("Done.")