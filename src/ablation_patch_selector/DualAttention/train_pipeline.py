"""Training pipeline for Raza dual attention on embedding inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, WeightedRandomSampler

repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.ablation_patch_selector.DualAttention.models import (
    RazaHardAttention,
    SoftAttentionEmbedding,
)
from src.ablation_patch_selector.DualAttention.sampling import (
    pairwise_distance_penalty,
    sample_attention_candidates,
)
from src.ablation_patch_selector.SASHA.data import (
    PTEmbeddingDirDataset,
    build_items_and_label_map,
    build_items_with_label_map,
    collate_samples,
)


@dataclass
class EpisodeRollout:
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    entropies: List[torch.Tensor]
    chosen_indices: List[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_float(value: float) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": _safe_float((y_true == y_pred).mean() if len(y_true) > 0 else 0.0),
        "f1": float("nan"),
        "auc": float("nan"),
        "num_cases": int(len(y_true)),
    }

    if len(y_true) == 0:
        return metrics

    try:
        from sklearn.metrics import f1_score, roc_auc_score

        metrics["f1"] = _safe_float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if y_prob.ndim == 2 and y_prob.size > 0:
            num_classes = int(y_prob.shape[1])
            if num_classes == 2:
                metrics["auc"] = _safe_float(roc_auc_score(y_true, y_prob[:, 1]))
            elif num_classes > 2:
                metrics["auc"] = _safe_float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )
    except Exception:
        pass

    return metrics


def _entropy(attn: torch.Tensor) -> torch.Tensor:
    attn = attn.clamp(min=1e-8)
    return -(attn * attn.log()).sum()


def _prepare_coords(coords: torch.Tensor, coord_dim: int) -> torch.Tensor:
    if coords is None or coords.numel() == 0:
        return torch.zeros(0, coord_dim, device=coords.device if coords is not None else None)
    if coords.dim() == 1:
        coords = coords.unsqueeze(0)
    if coords.shape[1] >= coord_dim:
        return coords[:, -coord_dim:]
    pad = torch.zeros(coords.shape[0], coord_dim - coords.shape[1], device=coords.device, dtype=coords.dtype)
    return torch.cat([coords, pad], dim=1)


def _run_hard_attention_episode(
    hard_model: RazaHardAttention,
    candidate_embeddings: torch.Tensor,
    candidate_coords: torch.Tensor,
    label_idx: int,
    num_glimpses: int,
    min_glimpse_dist: float,
    policy_entropy_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, EpisodeRollout, torch.Tensor, torch.Tensor]:
    device = candidate_embeddings.device
    num_candidates = int(candidate_embeddings.shape[0])

    if num_candidates <= 0:
        logits = torch.zeros(1, hard_model.num_classes, device=device)
        return (
            logits,
            torch.tensor(0.0, device=device),
            EpisodeRollout([], [], [], []),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        )

    num_glimpses = min(int(num_glimpses), num_candidates)

    cand = candidate_embeddings.unsqueeze(0)
    coords = candidate_coords.unsqueeze(0)
    cand_feat = hard_model.build_candidate_features(cand, coords)

    hidden, cell = hard_model.init_state(cand, coords)
    selected_mask = torch.zeros(1, num_candidates, dtype=torch.bool, device=device)

    rollout = EpisodeRollout([], [], [], [])

    for _ in range(num_glimpses):
        logits = hard_model.policy_logits(hidden, cand_feat, selected_mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        rollout.log_probs.append(dist.log_prob(action))
        rollout.entropies.append(dist.entropy())
        rollout.values.append(hard_model.value(hidden))

        new_mask = selected_mask.clone()
        new_mask.scatter_(1, action.unsqueeze(1), True)
        selected_mask = new_mask
        idx = int(action.item())
        rollout.chosen_indices.append(idx)

        embed_t = cand[:, idx, :]
        coord_t = coords[:, idx, :]
        hidden, cell = hard_model.step(hidden, cell, embed_t, coord_t)

    logits = hard_model.classify(hidden)
    target = torch.tensor([int(label_idx)], dtype=torch.long, device=device)
    class_loss = F.cross_entropy(logits, target)

    pred = int(torch.argmax(logits, dim=-1).item())
    reward = torch.tensor(1.0 if pred == int(label_idx) else 0.0, device=device)

    if rollout.values:
        values = torch.stack(rollout.values)
        log_probs = torch.stack(rollout.log_probs)
        entropies = torch.stack(rollout.entropies)
        advantages = reward - values.detach()
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, reward.expand_as(values))
        entropy_loss = -policy_entropy_weight * entropies.mean()
    else:
        policy_loss = torch.tensor(0.0, device=device)
        value_loss = torch.tensor(0.0, device=device)
        entropy_loss = torch.tensor(0.0, device=device)

    if rollout.chosen_indices:
        chosen_coords = candidate_coords[torch.tensor(rollout.chosen_indices, device=device)]
    else:
        chosen_coords = candidate_coords[:0]

    dist_penalty = pairwise_distance_penalty(chosen_coords, min_dist=min_glimpse_dist)

    hard_loss = class_loss + policy_loss + value_loss + entropy_loss + dist_penalty
    return logits, hard_loss, rollout, reward, dist_penalty


def _predict_greedy(
    hard_model: RazaHardAttention,
    candidate_embeddings: torch.Tensor,
    candidate_coords: torch.Tensor,
    num_glimpses: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    device = candidate_embeddings.device
    num_candidates = int(candidate_embeddings.shape[0])

    if num_candidates <= 0:
        logits = torch.zeros(1, hard_model.num_classes, device=device)
        return logits, torch.softmax(logits, dim=-1), []

    num_glimpses = min(int(num_glimpses), num_candidates)

    cand = candidate_embeddings.unsqueeze(0)
    coords = candidate_coords.unsqueeze(0)
    cand_feat = hard_model.build_candidate_features(cand, coords)
    hidden, cell = hard_model.init_state(cand, coords)

    selected_mask = torch.zeros(1, num_candidates, dtype=torch.bool, device=device)
    chosen = []

    for _ in range(num_glimpses):
        logits = hard_model.policy_logits(hidden, cand_feat, selected_mask)
        action = torch.argmax(logits, dim=-1)
        new_mask = selected_mask.clone()
        new_mask.scatter_(1, action.unsqueeze(1), True)
        selected_mask = new_mask
        idx = int(action.item())
        chosen.append(idx)

        embed_t = cand[:, idx, :]
        coord_t = coords[:, idx, :]
        hidden, cell = hard_model.step(hidden, cell, embed_t, coord_t)

    logits = hard_model.classify(hidden)
    probs = torch.softmax(logits, dim=-1)
    return logits, probs, chosen


def train_raza(args: argparse.Namespace) -> Dict[str, object]:
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items, label_map = build_items_and_label_map(args.train_embeddings_dir, input_format=args.input_format)
    train_items = items
    val_items = build_items_with_label_map(
        args.val_embeddings_dir,
        label_map=label_map,
        strict=False,
        input_format=args.input_format,
    )

    train_ds = PTEmbeddingDirDataset(
        train_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode=args.patch_sample_mode,
        sample_seed=args.seed,
    )
    val_ds = PTEmbeddingDirDataset(
        val_items,
        max_patches_per_wsi=args.max_patches_per_wsi,
        sample_mode=args.patch_sample_mode,
        sample_seed=args.seed,
    )

    train_labels = [int(item[2]) for item in train_items]
    class_counts = np.bincount(train_labels, minlength=len(label_map)).astype(np.float32)
    class_counts[class_counts == 0.0] = 1.0
    class_weights_arr = (len(train_labels) / (len(label_map) * class_counts)).astype(np.float32)
    sample_weights = torch.tensor([class_weights_arr[int(item[2])] for item in train_items], dtype=torch.float32)
    train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_samples)

    soft_model = SoftAttentionEmbedding(embed_dim=args.embed_dim, hidden_dim=args.soft_hidden_dim).to(device)
    hard_model = RazaHardAttention(
        embed_dim=args.embed_dim,
        coord_dim=args.coord_dim,
        hidden_dim=args.hard_hidden_dim,
        num_classes=len(label_map),
    ).to(device)
    hard_model.set_logit_adjustment(torch.tensor(class_counts, device=device), tau=args.logit_adjust_tau)

    params = list(soft_model.parameters()) + list(hard_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_path = None
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        soft_model.train()
        hard_model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            for sample in batch:
                embeddings = sample.embeddings.to(device)
                coords = _prepare_coords(sample.coords.to(device), args.coord_dim)
                n = int(embeddings.shape[0])
                if n <= 0:
                    continue

                optimizer.zero_grad()

                mask = torch.ones(1, n, dtype=torch.bool, device=device)
                attn = soft_model(embeddings.unsqueeze(0), mask).squeeze(0)
                soft_loss = -args.soft_entropy_beta * _entropy(attn)

                cand_emb, cand_coords, _ = sample_attention_candidates(
                    embeddings,
                    coords,
                    attn,
                    num_tiles=args.num_tiles,
                    pool_multiplier=args.pool_multiplier,
                    noise_low=args.noise_low,
                    noise_high=args.noise_high,
                    min_dist=args.min_tile_dist,
                )

                logits, hard_loss, _, _, _ = _run_hard_attention_episode(
                    hard_model,
                    cand_emb,
                    cand_coords,
                    sample.label_idx,
                    num_glimpses=args.num_glimpses,
                    min_glimpse_dist=args.min_glimpse_dist,
                    policy_entropy_weight=args.policy_entropy_weight,
                )

                soft_weight = args.soft_weight * (args.soft_decay ** epoch)
                loss = hard_loss + soft_weight * soft_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()

                epoch_loss += float(loss.item())
                epoch_steps += 1

        val_metrics = evaluate_raza(
            soft_model,
            hard_model,
            val_loader,
            device=device,
            args=args,
        )

        avg_loss = epoch_loss / max(1, epoch_steps)
        record = {"epoch": float(epoch), "train_loss": float(avg_loss)}
        record.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        history.append(record)

        val_acc = float(val_metrics.get("accuracy", 0.0))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = str(Path(args.out_dir) / "best_raza_dual_attention.pt")
            torch.save(
                {
                    "soft_state_dict": soft_model.state_dict(),
                    "hard_state_dict": hard_model.state_dict(),
                    "label_map": label_map,
                    "embed_dim": args.embed_dim,
                    "soft_hidden_dim": args.soft_hidden_dim,
                    "hard_hidden_dim": args.hard_hidden_dim,
                    "coord_dim": args.coord_dim,
                    "num_classes": len(label_map),
                    "num_glimpses": args.num_glimpses,
                    "num_tiles": args.num_tiles,
                },
                best_path,
            )

        if args.log_every > 0 and (epoch + 1) % args.log_every == 0:
            print(
                f"Epoch {epoch + 1}/{args.epochs} - train_loss={avg_loss:.4f} "
                f"val_acc={val_metrics.get('accuracy', 0.0):.4f}"
            )

    history_path = str(Path(args.out_dir) / "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return {
        "label_map": label_map,
        "best_path": best_path,
        "history_path": history_path,
        "best_val_acc": best_val_acc,
    }


def evaluate_raza(
    soft_model: SoftAttentionEmbedding,
    hard_model: RazaHardAttention,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    soft_model.eval()
    hard_model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                embeddings = sample.embeddings.to(device)
                coords = _prepare_coords(sample.coords.to(device), args.coord_dim)
                n = int(embeddings.shape[0])
                if n <= 0:
                    continue

                mask = torch.ones(1, n, dtype=torch.bool, device=device)
                attn = soft_model(embeddings.unsqueeze(0), mask).squeeze(0)

                cand_emb, cand_coords, _ = sample_attention_candidates(
                    embeddings,
                    coords,
                    attn,
                    num_tiles=args.num_tiles,
                    pool_multiplier=args.pool_multiplier,
                    noise_low=0.0,
                    noise_high=0.0,
                    min_dist=args.min_tile_dist,
                )

                logits, probs, _ = _predict_greedy(
                    hard_model,
                    cand_emb,
                    cand_coords,
                    num_glimpses=args.num_glimpses,
                )

                pred = int(torch.argmax(logits, dim=-1).item())
                y_true.append(int(sample.label_idx))
                y_pred.append(pred)
                y_prob.append(probs.squeeze(0).detach().cpu().tolist())

    return _compute_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Raza dual attention model")

    parser.add_argument("--train-embeddings-dir", default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/train", help="Directory with per-WSI .pt files containing training embeddings")
    parser.add_argument("--val-embeddings-dir", default="/Volumes/Xbox_HD/Data/extracted_with_embeddings/full/val", help="Directory with per-WSI .pt files containing validation embeddings")
    parser.add_argument("--out-dir", default="data/models/ablation_patch_selector/dual_attention")
    parser.add_argument(
        "--input-format",
        type=str,
        default="pt",
        choices=["pt", "svs", "auto"],
        help="Input file type expected in train/val directories.",
    )

    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--soft-hidden-dim", type=int, default=256)
    parser.add_argument("--hard-hidden-dim", type=int, default=256)
    parser.add_argument("--coord-dim", type=int, default=2)

    parser.add_argument("--num-tiles", type=int, default=12)
    parser.add_argument("--num-glimpses", type=int, default=6)
    parser.add_argument("--pool-multiplier", type=int, default=2)
    parser.add_argument("--noise-low", type=float, default=0.0)
    parser.add_argument("--noise-high", type=float, default=0.1)
    parser.add_argument("--min-tile-dist", type=float, default=0.0)
    parser.add_argument("--min-glimpse-dist", type=float, default=0.0)

    parser.add_argument("--soft-entropy-beta", type=float, default=0.1)
    parser.add_argument("--policy-entropy-weight", type=float, default=0.01)
    parser.add_argument("--soft-weight", type=float, default=1.0)
    parser.add_argument("--soft-decay", type=float, default=0.98)

    parser.add_argument("--max-patches-per-wsi", type=int, default=0)
    parser.add_argument("--patch-sample-mode", type=str, default="uniform", choices=["uniform", "random", "head"])

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--logit-adjust-tau", type=float, default=1.0,
                        help="Logit adjustment tau (0 = disabled)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    result = train_raza(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
