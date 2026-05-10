"""Module for wsi classification plip."""

# ============================================================
# PLIP WSI-level Diagnosis Baseline (Attention MIL)
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` package imports work when running
# this script directly
repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ---- import your existing code
from src.utils.wsi import WSI
from src.utils.embedder import Embedder
import json
import os
from typing import Dict, List, Tuple
import time


# ============================================================
# 1. Transformer-based MIL model
# ============================================================


class PositionalEncoding(nn.Module):
    """Add learnable positional encodings to patch embeddings."""

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model] or [seq_len, d_model]
        """
        if x.dim() == 2:
            # Single WSI case: [N, d_model]
            return x + self.pos_embedding[0, : x.size(0), :]
        else:
            # Batched case: [batch, N, d_model]
            return x + self.pos_embedding[:, : x.size(1), :]


class TransformerMIL(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        hidden_dim=256,
        num_classes=2,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Project input embeddings to hidden dimension
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        # Positional encoding for spatial context
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        x: Tensor [N, 512]  (patch embeddings for ONE WSI)
        Returns: logits [num_classes], attention_weights [num_layers, num_heads, N+1, N+1]
        """
        # Project to hidden dimension: [N, hidden_dim]
        x = self.input_proj(x)

        # Add positional encoding: [N, hidden_dim]
        x = self.pos_encoding(x)

        # Prepend CLS token: [1, hidden_dim] -> [N+1, hidden_dim]
        cls_tokens = self.cls_token.expand(1, -1, -1)  # [1, 1, hidden_dim]
        x = x.unsqueeze(0)  # [1, N, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [1, N+1, hidden_dim]

        # Apply transformer encoder
        # Note: we'll extract attention from the last layer for visualization
        encoded = self.transformer_encoder(x)  # [1, N+1, hidden_dim]

        # Extract CLS token representation
        cls_output = encoded[:, 0, :]  # [1, hidden_dim]

        # Classification
        logits = self.classifier(cls_output)  # [1, num_classes]
        logits = logits.squeeze(0)  # [num_classes]

        # Extract attention weights from first layer for visualization
        # This is a simplified version - full attention extraction would require hooks
        attn_weights = self._get_attention_weights(x)

        return logits, attn_weights

    def _get_attention_weights(self, x):
        """
        Extract attention weights for visualization.
        Returns average attention from CLS token to all patches.
        """
        # For visualization, we'll return a simplified attention map
        # In practice, you'd use hooks to extract actual attention weights
        # For now, return uniform attention as placeholder
        num_patches = x.size(1) - 1  # Exclude CLS token
        # Return attention from CLS to patches: [num_patches]
        return torch.ones(num_patches, 1, device=x.device) / num_patches


# ============================================================
# 2. Extract ALL PLIP embeddings from a WSI
# ============================================================


def extract_embeddings_for_wsi(image_path: str, skip_level_thresh: int = 40000):
    """Extract PLIP embeddings for one WSI and return a dict for saving.

    Skips any pyramid level whose width or height exceeds `skip_level_thresh`.
    """
    embedder = Embedder(img_backend="plip")
    wsi = WSI(image_path=image_path, embedder=embedder)

    embs = []
    levels_processed = []
    n_patches_per_level = {}

    for lvl_id in sorted(wsi.levels_info.keys()):
        # per-level guard
        try:
            lvl_w, lvl_h = wsi.levels_info[lvl_id]["size"]
            if lvl_w > skip_level_thresh or lvl_h > skip_level_thresh:
                print(
                    f"Skipping level {lvl_id} for {image_path}: {lvl_w}x{lvl_h} > {skip_level_thresh}"
                )
                continue
        except Exception:
            pass

        count = 0
        for x, y in wsi.iterate_patches(lvl_id):
            try:
                patch = wsi.get_patch(lvl_id, x, y)
                emb = wsi.get_emb(patch)
                # Ensure CPU tensor
                if hasattr(emb, "cpu"):
                    emb = emb.cpu()
                embs.append(emb)
                count += 1
            except Exception:
                continue

        if count > 0:
            levels_processed.append(lvl_id)
            n_patches_per_level[lvl_id] = count

    if len(embs) == 0:
        embeddings = torch.zeros(1, 512)
    else:
        embeddings = torch.stack(embs)

    return {
        "embeddings": embeddings,
        "levels": levels_processed,
        "n_patches_per_level": n_patches_per_level,
    }


# ============================================================
# 3. WSI-level Dataset
# ============================================================


class WSIDataset(Dataset):
    def __init__(self, items):
        """
        items: list of dicts
            {
              "image_path": str,
              "label": int
            }
        """
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    # batch size is 1 WSI → return raw dict
    return batch


# ============================================================
# 4. Training loop
# ============================================================


def train_plip_wsi_mil(
    wsi_items,
    num_classes,
    epochs=10,
    lr=1e-4,
    device="cuda",
    use_preextracted: bool = True,
    preextracted_dir: str = "data/embeddings/plip",
    force_extract: bool = False,
):
    dataset = WSIDataset(wsi_items)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    model = TransformerMIL(
        embed_dim=512,
        hidden_dim=256,
        num_classes=num_classes,
        num_heads=8,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(
        f"Start training: {len(wsi_items)} WSIs, num_classes={num_classes}, epochs={epochs}, device={device}"
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        total_batches = len(loader)
        pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="wsi")
        for batch_idx, batch in enumerate(pbar):
            item = batch[0]

            case_id = os.path.splitext(os.path.basename(item["image_path"]))[0]
            X = None

            if use_preextracted and not force_extract:
                pt_path = os.path.join(preextracted_dir, f"{case_id}.pt")
                try:
                    t0 = time.time()
                    loaded = torch.load(pt_path, map_location=device)
                    t_load = time.time() - t0
                    # support either tensor directly or dict with 'embeddings'
                    if isinstance(loaded, dict) and "embeddings" in loaded:
                        X = loaded["embeddings"].to(device)
                    elif isinstance(loaded, torch.Tensor):
                        X = loaded.to(device)
                    else:
                        # unexpected format
                        print(
                            f"Preextracted file {pt_path} has unexpected format, falling back to extraction"
                        )
                        X = None

                    if X is not None:
                        print(
                            f"Loaded preextracted embeddings for {case_id} ({X.shape[0]} patches) in {t_load:.3f}s"
                        )
                except Exception:
                    # missing or failed load -> fallback to extraction
                    print(
                        f"Preextracted embeddings not found or failed for {case_id}, extracting on-the-fly"
                    )

            if X is None:
                # ---- build WSI with PLIP embedder
                embedder = Embedder(img_backend="plip")
                wsi = WSI(image_path=item["image_path"], embedder=embedder)

                # ---- extract ALL embeddings (timed)
                t0 = time.time()
                print("Extracting PLIP embeddings...")
                # Use the local extractor that accepts an image path and returns a dict
                data = extract_embeddings_for_wsi(
                    item["image_path"]
                )  # {'embeddings': Tensor, ...}
                X = data["embeddings"].to(device)
                t_elapsed = time.time() - t0
                print(f"Extracted {X.shape[0]} patches in {t_elapsed:.3f}s")
            y = torch.tensor([item["label"]], device=device)

            optimizer.zero_grad()
            logits, attn = model(X)
            loss = criterion(logits.unsqueeze(0), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / (batch_idx + 1)
            lr_now = (
                optimizer.param_groups[0]["lr"]
                if "lr" in optimizer.param_groups[0]
                else lr
            )
            pbar.set_postfix(
                {
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{lr_now:.6f}",
                }
            )

            # Additional stdout logging for visibility
            print(
                f"Epoch {epoch} [{batch_idx+1}/{total_batches}]",
                f"image={item.get('image_path', '')}",
                f"batch_loss={loss.item():.4f}",
                f"avg_loss={avg_loss:.4f}",
                f"lr={lr_now:.6f}",
            )

        print(f"Epoch {epoch} completed | Avg loss = {avg_loss:.4f}")

    print("Training completed")

    return model


# ============================================================
# 5. Inference (with attention output)
# ============================================================


@torch.no_grad()
def infer_wsi(model, image_path, device="cuda"):
    model.eval()

    embedder = Embedder(img_backend="plip")
    wsi = WSI(image_path=image_path, embedder=embedder)
    # Use the extractor helper to support level skipping and consistent format
    data = extract_embeddings_for_wsi(image_path)
    X = data["embeddings"].to(device)
    logits, attn = model(X)

    probs = torch.softmax(logits, dim=-1)

    return {
        "logits": logits.cpu(),
        "probs": probs.cpu(),
        "prediction": probs.argmax().item(),
        "attention": attn.squeeze().cpu(),
    }


# ============================================================
# 6. Example usage
# ============================================================

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    import argparse

    def load_labels_json(labels_json_path: str) -> Dict[str, str]:
        """Load mapping file of case_id -> project_id."""
        with open(labels_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_wsi_items_from_labels(
        labels: Dict[str, str], images_dir: str = "data/images"
    ) -> Tuple[List[dict], Dict[str, int]]:
        """
        Convert a mapping of case_id -> project_id into a list of items
        expected by the training function: [{"image_path":..., "label": int}, ...]

        Returns items and the project->int label map.
        """
        # Collect unique projects and assign integer labels
        projects = sorted({p for p in labels.values() if p is not None})
        proj2int = {p: i for i, p in enumerate(projects)}

        items = []
        for case_id, proj in labels.items():
            img_path = os.path.join(images_dir, f"{case_id}.svs")
            if proj is None:
                # skip unlabeled or failed lookups
                continue
            label = proj2int[proj]
            items.append({"image_path": img_path, "label": label})

        return items, proj2int

    parser = argparse.ArgumentParser(
        description="Train / run WSI-level PLIP classifier"
    )
    parser.add_argument(
        "--labels-json",
        default="data/images/metadata/labels.json",
        help="Mapping file (case_id -> project_id)",
    )
    parser.add_argument(
        "--images-dir", default="data/images", help="Directory with .svs images"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--no-preextracted",
        action="store_true",
        help="Do not use preextracted embeddings; extract on-the-fly",
    )
    parser.add_argument(
        "--preex-dir",
        default="data/embeddings/plip",
        help="Directory containing preextracted .pt files",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force extraction even if preextracted file exists",
    )
    args = parser.parse_args()

    labels_path = args.labels_json
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = load_labels_json(labels_path)
    wsi_items, proj2int = build_wsi_items_from_labels(
        labels, images_dir=args.images_dir
    )

    model = train_plip_wsi_mil(
        wsi_items=wsi_items,
        num_classes=len(proj2int),
        epochs=args.epochs,
        device=args.device,
        use_preextracted=(not args.no_preextracted),
        preextracted_dir=args.preex_dir,
        force_extract=args.force_extract,
    )

    # Save trained model
    save_dir = os.path.join("data", "models", "downstream_tasks", "wsi_classification")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "attention_plip.pth")
    try:
        torch.save(model.state_dict(), save_path)
        print(f"Saved trained TransformerMIL model to: {save_path}")
    except Exception as e:
        print(f"Failed to save model to {save_path}: {e}")

    # Example inference on first item
    if len(wsi_items) > 0:
        out = infer_wsi(model, wsi_items[0]["image_path"], device=args.device)
        print("Prediction:", out["prediction"], "Probs:", out["probs"])
