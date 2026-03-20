# Ensure repo root is on sys.path so `src` package imports work when running
# this script directly (python src/training/zoom_classifier.py)
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# =========================
# MODEL
# =========================


class ZoomClassifier(nn.Module):
    """
    Binary classifier:
    output logits for [STOP, ZOOM]
    """

    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# TRAINING
# =========================


def train(
    device=None,
    epochs=100,
    batch_size=64,
    val_ratio=0.2,
    data_npz="data/supervised_dataset/score_regressor.npz",
    model_out="data/models/zoom_classifier.pth",
    random_seed=42,
):
    print("Starting training of Zoom Classifier...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # -------- Load dataset --------
    with np.load(data_npz) as d:
        states = torch.tensor(d["states"], dtype=torch.float32)
        zoom_labels = torch.tensor(d["zoom_decision"], dtype=torch.long)
        # zoom_labels ∈ {0,1}

    dataset = TensorDataset(states, zoom_labels)

    # -------- Train / Val split --------
    n_total = len(dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: {n_train} train / {n_val} val")

    # -------- Model --------
    model = ZoomClassifier(state_dim=states.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # =========================
    # TRAIN LOOP
    # =========================
    for ep in range(epochs):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += y.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)

                val_loss += loss.item()
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(
            f"[Z] Epoch {ep+1:3d}/{epochs} | "
            f"Train CE {train_loss:.4f} | Train Acc {train_acc:.3f} | "
            f"Val CE {val_loss:.4f} | Val Acc {val_acc:.3f}"
        )

    # -------- Save --------
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    torch.save(model.state_dict(), model_out)

    return model


# =========================
# CLI
# =========================


def _cli():
    p = argparse.ArgumentParser(description="Train supervised Zoom classifier")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.2)
    return p


if __name__ == "__main__":
    train()


# Result
# [Z] Epoch 100/100 | Train CE 0.1683 | Train Acc 0.929 | Val CE 0.1864 | Val Acc 0.921
