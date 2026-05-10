"""
Concrete training example: 5 WSIs with different patch counts.

Step 1 (run once):
    python src/downstream_task/extract_method_embeddings.py --method humbe

Step 2 (this script):
    python src/downstream_task/train_example.py
"""

import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.downstream_task.wsi_classification_plip import TransformerMIL

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "/Volumes/Xbox_HD/Data/downstream_data/humbe"  # .pt files from extract_method_embeddings.py
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 5 WSIs: (case_id, integer_label)
# Labels: LUAD=0, LUSC=1
CASES = [
    ("TCGA-05-4244-LUAD", 0),  # N patches (varies)
    ("TCGA-05-4245-LUAD", 0),
    ("TCGA-05-4249-LUAD", 0),
    ("TCGA-18-3406-LUSC", 1),
    ("TCGA-18-3407-LUSC", 1),
]
NUM_CLASSES = 2

# ─────────────────────────────────────────────────────────────
# Load pre-extracted embeddings into memory
# Each WSI has a DIFFERENT N — we keep them as a plain list,
# never stack them, because N varies.
# ─────────────────────────────────────────────────────────────
dataset = []
for case_id, label in CASES:
    pt_path = os.path.join(EMBEDDINGS_DIR, f"{case_id}.pt")
    if not os.path.exists(pt_path):
        print(f"[SKIP] {pt_path} not found")
        continue
    data = torch.load(pt_path, map_location="cpu")
    embeddings = data["embeddings"]  # Tensor[N, 512]  — N differs per WSI
    dataset.append((embeddings, label, case_id))
    print(f"  Loaded {case_id}: {embeddings.shape[0]} patches, label={label}")

if not dataset:
    raise RuntimeError("No .pt files found — run extract_method_embeddings.py first.")

# ─────────────────────────────────────────────────────────────
# Model, optimizer, loss
# ─────────────────────────────────────────────────────────────
model = TransformerMIL(
    embed_dim=512,
    hidden_dim=256,
    num_classes=NUM_CLASSES,
    num_heads=8,
    num_layers=2,
    dropout=0.1,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────
# Training loop — batch_size=1 per WSI (variable-length N)
# ─────────────────────────────────────────────────────────────
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for embeddings, label, case_id in dataset:
        X = embeddings.to(DEVICE)  # [N, 512]   N varies per WSI
        y = torch.tensor([label], device=DEVICE)  # [1]

        optimizer.zero_grad()
        logits, _ = model(X)  # logits: [num_classes]
        loss = criterion(logits.unsqueeze(0), y)  # unsqueeze → [1, num_classes]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(
            f"  Epoch {epoch+1} | {case_id} ({embeddings.shape[0]} patches) | loss={loss.item():.4f}"
        )

    print(f"Epoch {epoch+1}/{EPOCHS} avg_loss={total_loss/len(dataset):.4f}\n")

# ─────────────────────────────────────────────────────────────
# Quick eval (no separate val split — just to show inference)
# ─────────────────────────────────────────────────────────────
model.eval()
label_names = {0: "LUAD", 1: "LUSC"}
print("─── Quick predictions ───")
with torch.no_grad():
    for embeddings, label, case_id in dataset:
        X = embeddings.to(DEVICE)
        logits, _ = model(X)
        pred = torch.softmax(logits, dim=-1).argmax().item()
        correct = "✓" if pred == label else "✗"
        print(
            f"  {correct} {case_id}: pred={label_names[pred]}  true={label_names[label]}"
        )
