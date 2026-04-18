import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from torch.utils.data import DataLoader, Dataset

# =========================
# PATHS
# =========================
DATA_DIR = r"/home/skabbo/datasets/cassini/Cassini Spacecraft/windows_2005_focus6h/scaled"
TRAIN_X = os.path.join(DATA_DIR, "train_X.npy")
TRAIN_Y = os.path.join(DATA_DIR, "train_y.npy")
VAL_X   = os.path.join(DATA_DIR, "val_X.npy")
VAL_Y   = os.path.join(DATA_DIR, "val_y.npy")
TEST_X  = os.path.join(DATA_DIR, "test_X.npy")
TEST_Y  = os.path.join(DATA_DIR, "test_y.npy")
CLASS_WEIGHTS_PATH = os.path.join(DATA_DIR, "class_weights.pkl")

# =========================
# TRAINING CONFIG
# =========================
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 1e-3
NUM_WORKERS = 0
DEVICE      = torch.device("cpu")

class WindowDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = torch.tensor(np.load(x_path), dtype=torch.float32)
        self.y = torch.tensor(np.load(y_path), dtype=torch.long)
        assert len(self.X) == len(self.y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BiLSTMClassifier(nn.Module):
    """
    Pure BiLSTM baseline for 2005 data.
    Input:  (B, T, F) where F=63
    Output: (B, 3)
    """
    def __init__(self, n_features=63, hidden=128, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        center = out[:, out.shape[1] // 2, :]
        return self.fc(center)

def make_loader(x_path, y_path, shuffle):
    return DataLoader(
        WindowDataset(x_path, y_path),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS
    )

def load_class_weights():
    w = joblib.load(CLASS_WEIGHTS_PATH)
    return torch.tensor([w[0], w[1], w[2]], dtype=torch.float32, device=DEVICE)

def evaluate(model, loader, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss_sum += criterion(logits, yb).item() * xb.shape[0]
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.shape[0]
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())
    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # per-class recall
    for cls, name in enumerate(['None', 'BS', 'MP']):
        mask = all_labels == cls
        if mask.sum() > 0:
            recall = (all_preds[mask] == cls).float().mean().item()
            print(f"  Recall {name}: {recall:.4f}  ({mask.sum().item()} samples)")

    return loss_sum / max(total, 1), correct / max(total, 1)

def main():
    print("Loading 2005 datasets...")
    train_loader = make_loader(TRAIN_X, TRAIN_Y, shuffle=True)
    val_loader   = make_loader(VAL_X,   VAL_Y,   shuffle=False)
    test_loader  = make_loader(TEST_X,  TEST_Y,  shuffle=False)

    print("Building BiLSTM model...")
    model = BiLSTMClassifier().to(DEVICE)
    class_weights = load_class_weights()
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        model.train()
        total_loss, total_n = 0.0, 0
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item() * xb.shape[0]
            total_n    += xb.shape[0]
            if step % 50 == 0:
                print(f"  step {step} loss {loss.item():.4f}")

        train_loss = total_loss / max(total_n, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch} TRAIN loss {train_loss:.4f}")
        print(f"Epoch {epoch} VAL   loss {val_loss:.4f}  acc {val_acc:.4f}")

        torch.save(model.state_dict(), f"model_2005_epoch{epoch}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_2005.pt")
            print("  --> saved best_model_2005.pt")

    print("\nFinal TEST evaluation...")
    model.load_state_dict(torch.load("best_model_2005.pt", map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"TEST loss {test_loss:.4f}  acc {test_acc:.4f}")

if __name__ == "__main__":
    main()
