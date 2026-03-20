# Ensure repo root is on sys.path so `src` package imports work when running
# this script directly (python src/training/supervised_score_regressor.py)
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


class ScoreRegressor(nn.Module):
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
def zoom_accuracy(pred, target):
    """
    pred, target: (B, 2)
    """
    pred_zoom = pred[:, 1] >= pred[:, 0]
    gt_zoom = target[:, 1] >= target[:, 0]
    return (pred_zoom == gt_zoom).float().mean().item()


def train(
    device=None,
    epochs=100,
    data_npz: str = None,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    model_out: str = "data/models/score_regressor.pth",
    random_seed: int = 42,
):
    print("Starting training of Score Regressor ...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # -------- Load dataset --------
    if data_npz is None:
        raise ValueError("data_npz must be provided.")
    with np.load(data_npz) as d:
        states = torch.tensor(d["states"], dtype=torch.float32)
        scores = torch.tensor(d["scores"], dtype=torch.float32)

    full_dataset = TensorDataset(states, scores)

    # -------- Train / Val split --------
    n_total = len(full_dataset)
    n_val = int(val_ratio * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: {n_train} train / {n_val} val")

    # -------- Model --------
    model = ScoreRegressor(state_dim=states.shape[1], hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # =========================
    # TRAIN LOOP
    # =========================
    for ep in range(epochs):
        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0
        val_zoom_acc = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)

                val_loss += loss_fn(pred, y).item()
                val_zoom_acc += zoom_accuracy(pred, y)
                n_batches += 1

        val_loss /= max(1, n_batches)
        val_zoom_acc /= max(1, n_batches)

        print(
            f"[G] Epoch {ep+1:3d}/{epochs} | "
            f"Train MSE {train_loss:.4f} | "
            f"Val MSE {val_loss:.4f} | "
            f"Zoom Acc {val_zoom_acc:.3f}"
        )

    # -------- Save --------
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_out)

    return model


# =========================
# CLI
# =========================


def _cli():
    p = argparse.ArgumentParser(description="Train supervised score regressor (G)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--model-out", type=str, required=True)
    p.add_argument("--data-npz", type=str, default=None)
    return p


if __name__ == "__main__":
    print("Training A1...")
    train(data_npz="data/supervised_dataset/a1.npz", model_out="data/models/supervised/score_regressor/a1.pth", epochs=100)
    print("\nTraining A2...")
    train(data_npz="data/supervised_dataset/a2.npz", model_out="data/models/supervised/score_regressor/a2.pth", epochs=100)
    print("\nTraining A3...")
    train(data_npz="data/supervised_dataset/a3.npz", model_out="data/models/supervised/score_regressor/a3.pth", epochs=100)
    print("\nTraining B1...")
    train(data_npz="data/supervised_dataset/b1.npz", model_out="data/models/supervised/score_regressor/b1.pth", epochs=100)
    print("\nTraining B2...")
    train(data_npz="data/supervised_dataset/b2.npz", model_out="data/models/supervised/score_regressor/b2.pth", epochs=100)
    print("\nTraining C1...")
    train(data_npz="data/supervised_dataset/c1.npz", model_out="data/models/supervised/score_regressor/c1.pth", epochs=100)
    print("\nTraining C2...")
    train(data_npz="data/supervised_dataset/c2.npz", model_out="data/models/supervised/score_regressor/c2.pth", epochs=100)
    print("\nTraining D1...")
    train(data_npz="data/supervised_dataset/d1.npz", model_out="data/models/supervised/score_regressor/d1.pth", epochs=100)
    print("\nTraining D2...")
    train(data_npz="data/supervised_dataset/d2.npz", model_out="data/models/supervised/score_regressor/d2.pth", epochs=100)
    print("\nTraining E1...")
    train(data_npz="data/supervised_dataset/e1.npz", model_out="data/models/supervised/score_regressor/e1.pth", epochs=100)
    print("\nTraining E2...")
    train(data_npz="data/supervised_dataset/e2.npz", model_out="data/models/supervised/score_regressor/e2.pth", epochs=100)


# Result
# [G] Epoch 100/100 | Train MSE 3.0380 | Val MSE 3.3584 | Zoom Acc 0.903
